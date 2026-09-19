#!/usr/bin/env python3
"""Protocol driver for one dataset x backbone cell.

  python scripts/run_cell.py --dataset Amazon-VG --backbone MF                 # full search + finals + test
  python scripts/run_cell.py --dataset Amazon-VG --backbone MF \
      --selected configs/selected_paper.json                                   # skip the search
  python scripts/run_cell.py --dataset Amazon-VG --backbone MF \
      --arms_file configs/ablations_vg.json --arms SSNS_only,fresh_only,shared_gain

Stages (all resume-safe; finished runs are skipped)
  0  SimGCL only: contrastive weight selected on the base model (grid in configs/grids.json)
  1  base finals, seed 20 (its checkpoint is the frozen backbone of the DDC search)
  2  base finals, seeds 21/22, and the validation search of every arm
     (seed 20, --sweep_epochs cap, validation every --eval_every epochs, --patience)
  3  candidate scoring: the run's best validation Recall@20; on Douban every candidate
     checkpoint is re-evaluated on the full validation set (training-time validation
     uses a fixed 4,000-user subset); PDA candidates are served with each alpha
  4  winner per arm (highest validation Recall@20; first in grid order on ties) -> selection.json
  5  finals: seeds --seeds with the --final_epochs cap; the seed-20 search run is reused
     when early stopping ended it before the cap; DDC finals use the seed-matched base
  6  test evaluation of every final run at K=20 (with the rank and top-20 list dumps)

Layout: <out>/<data>/<backbone>/{lamsel,sweep,final}/..., selection.json, summary.csv
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        s = '%g' % v
        return s
    return str(v)


def slug(cfg):
    return '_'.join('%s%s' % (k, fmt(v)) for k, v in sorted(cfg.items()))


def run_name(arm, cfg, seed=None):
    s = arm + ('_' + slug(cfg) if cfg else '')
    return s + ('_s%d' % seed if seed is not None else '')


def read_json(path):
    with open(path) as f:
        return json.load(f)


class Cell:
    def __init__(self, a):
        self.a = a
        self.data_name = a.data_dir or a.dataset
        self.cell_dir = os.path.join(a.out, self.data_name, a.backbone)
        os.makedirs(self.cell_dir, exist_ok=True)
        grids = read_json(os.path.join(ROOT, 'configs', 'grids.json'))
        self.lambda_grid = grids['simgcl_lambda']
        self.arms = dict(grids['arms'])
        if a.arms_file:
            self.arms.update(read_json(a.arms_file)['arms'])
        names = a.arms.split(',') if a.arms else list(grids['arms'].keys())
        unknown = [n for n in names if n not in self.arms]
        if unknown:
            raise SystemExit('unknown arms: %s' % unknown)
        self.arm_names = names
        self.seeds = [int(s) for s in a.seeds.split(',')]
        self.selected = None
        if a.selected:
            key = '%s/%s' % (self.data_name, a.backbone)
            sel = read_json(a.selected)
            if key not in sel:
                raise SystemExit('%s has no entry for %s' % (a.selected, key))
            self.selected = sel[key]
        self.douban = a.dataset.startswith('Douban')
        self.eval_users = a.eval_users if a.eval_users is not None else (4000 if self.douban else 0)
        self.reeval_val = a.reeval_val if a.reeval_val is not None else self.douban
        self.simgcl_lambda = a.simgcl_lambda
        if self.simgcl_lambda is None and self.selected and self.selected.get('simgcl_lambda') is not None:
            self.simgcl_lambda = self.selected['simgcl_lambda']
        self.selection_path = os.path.join(self.cell_dir, 'selection.json')
        self.selection = read_json(self.selection_path) if os.path.exists(self.selection_path) else {}

    # ---------------------------------------------------------------- commands
    def train_cmd(self, run_dir, method, seed, epochs, cfg, flags=()):
        a = self.a
        cmd = [PY, '-m', 'stfr.train', '--dataset', a.dataset, '--backbone', a.backbone, '--method', method,
               '--seed', str(seed), '--epochs', str(epochs), '--eval_every', str(a.eval_every),
               '--patience', str(a.patience), '--eval_users', str(self.eval_users),
               '--run_dir', run_dir, '--device', a.device]
        if a.data_dir:
            cmd += ['--data_dir', a.data_dir]
        if a.backbone == 'SimGCL':
            cmd += ['--simgcl_lambda', fmt(self.simgcl_lambda)]
        for k, v in sorted(cfg.items()):
            cmd += ['--' + k, fmt(v)]
        cmd += list(flags)
        return cmd

    def eval_cmd(self, run_dir, split, k, dumps=False, eval_users=0):
        cmd = [PY, '-m', 'stfr.eval_ckpt', '--run_dir', run_dir, '--split', split, '--topk', str(k),
               '--device', self.a.device, '--eval_users', str(eval_users)]
        if dumps:
            cmd += ['--dump_recs']
        return cmd

    @staticmethod
    def train_done(run_dir):
        return os.path.exists(os.path.join(run_dir, 'metrics.json'))

    @staticmethod
    def eval_done(run_dir, split, k, dumps=False):
        """Metrics written and, for a run evaluated with dumps, its non-empty top-K list dump."""
        if not os.path.exists(os.path.join(run_dir, '%s_k%d.json' % (split, k))):
            return False
        if dumps:
            p = os.path.join(run_dir, '%s_recs_k%d.txt' % (split, k))
            return os.path.exists(p) and os.path.getsize(p) > 0
        return True

    def run_stage(self, name, jobs):
        """jobs: list of (label, [cmd, ...]); each job runs its commands sequentially."""
        jobs = [j for j in jobs if j[1]]
        print('[%s] %d job(s)' % (name, len(jobs)), flush=True)
        if self.a.dry_run:
            for label, cmds in jobs:
                for c in cmds:
                    print('  ', label, ':', ' '.join(c))
            return

        def work(job):
            label, cmds = job
            for c in cmds:
                run_dir = c[c.index('--run_dir') + 1]
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, 'stdout.log'), 'a') as out:
                    r = subprocess.run(c, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
                if r.returncode != 0:
                    raise RuntimeError('%s failed (%s); see %s/stdout.log' % (label, ' '.join(c), run_dir))
            print('  done:', label, flush=True)

        with cf.ThreadPoolExecutor(max_workers=self.a.jobs) as ex:
            for _ in ex.map(work, jobs):
                pass

    # ---------------------------------------------------------------- stages
    def stage_lambda(self):
        if self.a.backbone != 'SimGCL' or self.simgcl_lambda is not None:
            return
        lam_path = os.path.join(self.cell_dir, 'lambda.json')
        if os.path.exists(lam_path):
            self.simgcl_lambda = read_json(lam_path)['simgcl_lambda']
            return
        jobs = []
        for L in self.lambda_grid:
            rd = os.path.join(self.cell_dir, 'lamsel', 'L%s' % fmt(L))
            if not self.train_done(rd):
                self.simgcl_lambda = L
                jobs.append(('lamsel L=%s' % fmt(L), [self.train_cmd(rd, 'base', 20, self.a.sweep_epochs, {})]))
        self.simgcl_lambda = None
        self.run_stage('stage 0: SimGCL lambda', jobs)
        if self.a.dry_run:
            self.simgcl_lambda = self.lambda_grid[0]
            return
        best, best_v = None, -1.0
        for L in self.lambda_grid:
            v = read_json(os.path.join(self.cell_dir, 'lamsel', 'L%s' % fmt(L), 'metrics.json'))['best']['recall']
            if v > best_v:
                best, best_v = L, v
        self.simgcl_lambda = best
        with open(lam_path, 'w') as f:
            json.dump(dict(simgcl_lambda=best, val_recall=best_v,
                           grid={fmt(L): read_json(os.path.join(self.cell_dir, 'lamsel', 'L%s' % fmt(L), 'metrics.json'))['best']['recall']
                                 for L in self.lambda_grid}), f, indent=2)
        print('[stage 0] simgcl_lambda -> %s (val recall %.5f)' % (fmt(best), best_v))

    def base_final_dir(self, seed):
        return os.path.join(self.cell_dir, 'final', run_name('base', {}, seed))

    def base_final_job(self, seed):
        rd = self.base_final_dir(seed)
        if self.train_done(rd):
            return None
        if seed == 20 and self.a.backbone == 'SimGCL' and self.simgcl_lambda is not None:
            src = os.path.join(self.cell_dir, 'lamsel', 'L%s' % fmt(self.simgcl_lambda))
            if self.reusable(src):
                shutil.copytree(src, rd)
                return None
        return ('base final s%d' % seed, [self.train_cmd(rd, 'base', seed, self.a.final_epochs, {})])

    @staticmethod
    def reusable(run_dir):
        """a search run that early stopping ended before its cap is its own seed-20 final."""
        log = os.path.join(run_dir, 'train.log')
        return os.path.exists(os.path.join(run_dir, 'metrics.json')) and \
            'early stop' in open(log, errors='ignore').read()

    def arm_cfgs(self, arm):
        """(fixed args, flags, grid) of an arm after resolving inheritance."""
        spec = self.arms[arm]
        fixed = {}
        if spec.get('inherit'):
            inh = self.winner_cfg(spec['inherit'])
            fixed.update(inh)
            for dst, srckey in spec.get('inherit_map', {}).items():
                fixed[dst] = inh[srckey]
        fixed.update(spec.get('args', {}))
        return fixed, tuple(spec.get('flags', ())), spec.get('grid')

    def winner_cfg(self, arm):
        if self.selected is not None and arm in self.selected:
            return dict(self.selected[arm])
        if arm in self.selection:
            return dict(self.selection[arm]['config'])
        if arm == 'base':
            return {}
        if self.a.dry_run:
            print('  [dry-run] no selection for %s yet; inherited arms shown with an empty configuration' % arm)
            return {}
        raise SystemExit('no selected configuration for arm %s (run the search or pass --selected)' % arm)

    def stage_sweeps(self, arm_names, with_base=True):
        jobs = []
        for seed in (self.seeds if with_base else []):
            if seed == 20:
                continue
            j = self.base_final_job(seed)
            if j:
                jobs.append(j)
        base20 = os.path.join(self.base_final_dir(20), 'best.pth')
        for arm in arm_names:
            if arm == 'base' or (self.selected is not None and arm in self.selected):
                continue
            fixed, flags, grid = self.arm_cfgs(arm)
            if not grid:
                continue
            method = self.arms[arm]['method']
            for g in grid:
                rd = os.path.join(self.cell_dir, 'sweep', run_name(arm, g))
                if self.train_done(rd):
                    continue
                cfg_run = dict(fixed, **g)
                if self.arms[arm].get('needs_base'):
                    cfg_run['ddc_ckpt'] = base20
                jobs.append(('sweep %s' % run_name(arm, g),
                             [self.train_cmd(rd, method, 20, self.a.sweep_epochs, cfg_run, flags)]))
        self.run_stage('stage 2: base finals s21/s22 + validation search', jobs)

    def stage_score(self, arm_names):
        jobs = []
        for arm in arm_names:
            if arm == 'base' or (self.selected is not None and arm in self.selected):
                continue
            fixed, flags, grid = self.arm_cfgs(arm)
            if not grid:
                continue
            serve = self.arms[arm].get('serve_grid')
            for g in grid:
                rd = os.path.join(self.cell_dir, 'sweep', run_name(arm, g))
                if serve:
                    for s in serve:
                        sd = os.path.join(rd, 'serve_' + slug(s))
                        if self.eval_done(sd, 'val', 20):
                            continue
                        self.make_serve_dir(rd, sd, s)
                        jobs.append(('serve %s/%s' % (run_name(arm, g), slug(s)), [self.eval_cmd(sd, 'val', 20)]))
                elif self.reeval_val and not self.eval_done(rd, 'val', 20):
                    jobs.append(('reeval %s' % run_name(arm, g), [self.eval_cmd(rd, 'val', 20)]))
        self.run_stage('stage 3: candidate scoring', jobs)

    @staticmethod
    def make_serve_dir(train_dir, serve_dir, serve_cfg):
        """a served variant of a trained run: same checkpoint, serving arguments changed."""
        os.makedirs(serve_dir, exist_ok=True)
        cfg = read_json(os.path.join(train_dir, 'config.json'))
        cfg.update(serve_cfg)
        cfg['run_dir'] = serve_dir
        with open(os.path.join(serve_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        dst = os.path.join(serve_dir, 'best.pth')
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(os.path.join(train_dir, 'best.pth')), dst)

    def candidate_score(self, rd):
        if self.reeval_val:
            return read_json(os.path.join(rd, 'val_k20.json'))['recall']
        return read_json(os.path.join(rd, 'metrics.json'))['best']['recall']

    def stage_select(self, arm_names):
        if self.a.dry_run:
            return
        for arm in arm_names:
            if arm == 'base' or (self.selected is not None and arm in self.selected):
                continue
            fixed, flags, grid = self.arm_cfgs(arm)
            if not grid:
                continue
            serve = self.arms[arm].get('serve_grid')
            best, best_g, scores = -1.0, None, {}
            for g in grid:
                rd = os.path.join(self.cell_dir, 'sweep', run_name(arm, g))
                if serve:
                    for s in serve:
                        sd = os.path.join(rd, 'serve_' + slug(s))
                        v = read_json(os.path.join(sd, 'val_k20.json'))['recall']
                        gs = dict(g, **s)
                        scores[slug(gs)] = v
                        if v > best:
                            best, best_g = v, gs
                else:
                    v = self.candidate_score(rd)
                    scores[slug(g)] = v
                    if v > best:
                        best, best_g = v, g
            self.selection[arm] = dict(config=dict(fixed, **best_g), grid_config=best_g,
                                       val_recall=best, candidates=scores)
            print('[stage 4] %s -> %s (val recall %.5f)' % (arm, slug(best_g), best))
        with open(self.selection_path, 'w') as f:
            json.dump(self.selection, f, indent=2)

    def final_cfg(self, arm):
        """(full training configuration, flags, searched values used in run names) or None."""
        if arm == 'base':
            return {}, (), {}
        fixed, flags, grid = self.arm_cfgs(arm)
        if self.selected is not None and arm in self.selected:
            return dict(fixed, **self.selected[arm]), flags, dict(self.selected[arm])
        if grid:
            if arm not in self.selection:
                return None
            return dict(self.selection[arm]['config']), flags, dict(self.selection[arm]['grid_config'])
        return fixed, flags, {}

    def stage_finals(self):
        jobs = []
        for arm in self.arm_names:
            if arm == 'base':
                continue
            method = self.arms[arm]['method']
            fc = self.final_cfg(arm)
            if fc is None:
                print('  [skip] %s: no selected configuration yet' % arm)
                continue
            cfg, flags, name_cfg = fc
            serve_keys = [k for s in (self.arms[arm].get('serve_grid') or []) for k in s]
            for seed in self.seeds:
                rd = os.path.join(self.cell_dir, 'final', run_name(arm, name_cfg, seed))
                if self.train_done(rd):
                    continue
                if seed == 20:
                    src = os.path.join(self.cell_dir, 'sweep',
                                       run_name(arm, {k: v for k, v in name_cfg.items() if k not in serve_keys}))
                    if self.reusable(src):
                        shutil.copytree(src, rd)
                        c = read_json(os.path.join(rd, 'config.json'))
                        c.update({k: cfg[k] for k in serve_keys})
                        c['run_dir'] = rd
                        with open(os.path.join(rd, 'config.json'), 'w') as f:
                            json.dump(c, f, indent=2, sort_keys=True)
                        continue
                run_cfg = dict(cfg)
                if self.arms[arm].get('needs_base'):
                    run_cfg['ddc_ckpt'] = os.path.join(self.base_final_dir(seed), 'best.pth')
                jobs.append(('final %s' % run_name(arm, name_cfg, seed),
                             [self.train_cmd(rd, method, seed, self.a.final_epochs, run_cfg, flags)]))
        self.run_stage('stage 5: finals', jobs)

    def final_dirs(self):
        out = []
        for arm in self.arm_names:
            fc = self.final_cfg(arm)
            if fc is None:
                continue
            cfg, _, name_cfg = fc
            for seed in self.seeds:
                rd = os.path.join(self.cell_dir, 'final', run_name(arm, name_cfg, seed))
                if os.path.isdir(rd) and not os.path.exists(os.path.join(rd, 'arm.json')):
                    with open(os.path.join(rd, 'arm.json'), 'w') as f:
                        json.dump(dict(arm=arm, method=self.arms[arm]['method'], config=cfg, seed=seed,
                                       dataset=self.a.dataset, data=self.data_name, backbone=self.a.backbone,
                                       simgcl_lambda=self.simgcl_lambda), f, indent=2)
                out.append((arm, seed, rd))
        return out

    def stage_test(self):
        jobs = []
        for arm, seed, rd in self.final_dirs():
            cmds = []
            # a run with metrics but no list dump is evaluated again from its saved checkpoint
            if not self.eval_done(rd, 'test', 20, dumps=True):
                cmds.append(self.eval_cmd(rd, 'test', 20, dumps=True))
            if cmds:
                jobs.append(('test %s s%d' % (arm, seed), cmds))
        self.run_stage('stage 6: test evaluation', jobs)

    def summary(self):
        if self.a.dry_run:
            return
        rows = ['arm,seed,split,k,recall,ndcg,nalrp,run_dir']
        for arm, seed, rd in self.final_dirs():
            for k in (20,):
                p = os.path.join(rd, 'test_k%d.json' % k)
                if os.path.exists(p):
                    r = read_json(p)
                    rows.append('%s,%d,test,%d,%.6f,%.6f,%.6f,%s' % (arm, seed, k, r['recall'], r['ndcg'], r['nalrp'], rd))
        with open(os.path.join(self.cell_dir, 'summary.csv'), 'w') as f:
            f.write('\n'.join(rows) + '\n')
        print('[summary] %s/summary.csv' % self.cell_dir)

    def run(self):
        self.stage_lambda()
        j = self.base_final_job(20)
        self.run_stage('stage 1: base final s20', [j] if j else [])
        # arms that inherit another arm's selected configuration are searched after it is known
        independent = [a for a in self.arm_names if not self.arms[a].get('inherit')]
        inherited = [a for a in self.arm_names if self.arms[a].get('inherit')]
        for i, group in enumerate((independent, inherited)):
            if not group:
                continue
            self.stage_sweeps(group, with_base=(i == 0))
            self.stage_score(group)
            self.stage_select(group)
        self.stage_finals()
        self.stage_test()
        self.summary()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True)
    p.add_argument('--data_dir', default=None, help='data/<data_dir> variant of the dataset')
    p.add_argument('--backbone', required=True, choices=('MF', 'LightGCN', 'SimGCL'))
    p.add_argument('--out', default='runs')
    p.add_argument('--arms', default=None, help='comma-separated arm names (default: all main arms)')
    p.add_argument('--arms_file', default=None, help='JSON with additional arm definitions')
    p.add_argument('--selected', default=None, help='JSON of selected configurations (skips the search)')
    p.add_argument('--seeds', default='20,21,22')
    p.add_argument('--jobs', type=int, default=1, help='concurrent runs')
    p.add_argument('--sweep_epochs', type=int, default=30)
    p.add_argument('--final_epochs', type=int, default=200)
    p.add_argument('--eval_every', type=int, default=3)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--eval_users', type=int, default=None, help='training-time validation subset (default 4000 on Douban)')
    p.add_argument('--reeval_val', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=None,
                   help='re-evaluate candidates on the full validation set (default: on Douban)')
    p.add_argument('--simgcl_lambda', type=float, default=None, help='skip the SimGCL lambda selection')
    p.add_argument('--device', default='auto')
    p.add_argument('--dry_run', action='store_true', help='print the commands of the next stages')
    a = p.parse_args()
    Cell(a).run()


if __name__ == '__main__':
    main()
