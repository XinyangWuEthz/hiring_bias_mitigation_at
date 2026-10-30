"""Existing-margin diagnostics only. Does not change monitor weights or gates.

Fixed ablations: 8 bins, 16 bins split at benchmark-parent medians,
condition and demographic strata, anchor-label strata, and equal candidate/pair
weight within bins. Arm outcomes never choose cuts or features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EDGES = np.array([-4.595, -2.197, -.847, 0., .847, 2.197, 4.595])
ROUNDED_HIRE = np.array([.2691, .0849, .0428, .0263, .0459, .057, .1016, .3725])
PARENT = 'margin__SFT-v3'
MAIN = ['DAPO-v3-all', 'GRPO-v3-all', 'GRPO-v3-all-step525']


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def summarize(frame, arm):
    p = frame[PARENT].to_numpy() > 0
    a = frame['margin__' + arm].to_numpy() > 0
    change = a.astype(float) - p
    loss, gain = p & ~a, ~p & a
    return dict(rows=len(frame), candidates=frame.candidate_id.nunique(),
                pairs=frame.pair_id.nunique(),
                parent_hire_share=float(p.mean()),
                loss_rows=int(loss.sum()), gain_rows=int(gain.sum()),
                loss_pp=float(loss.mean()*100), gain_pp=float(gain.mean()*100),
                delta_pp=float(change.mean()*100),
                parent_margin_quantiles=np.quantile(frame[PARENT], [0, .25, .5, .75, 1]),
                ref_hire_share=float(frame.ref_hire.mean()))


def key_tuples(frame, columns):
    return list(frame[columns].itertuples(index=False, name=None))


def estimator(monitor, target, arms, columns, reference=False, equal_unit=None):
    """Poststratification diagnostic, target already restricted to reference hire.

    Unknown cells are not set to zero or silently renormalized: return known
    contribution, supported-mass-normalized result and worst-case bounds.
    """
    m = monitor[monitor.ref_hire].copy() if reference else monitor.copy()
    mk = key_tuples(m, columns)
    tk = key_tuples(target, columns)
    masses = pd.Series(tk, dtype=object).value_counts().to_dict()
    masses = {k: v/len(target) for k, v in masses.items()}
    positions = {}
    for i, key in enumerate(mk):
        positions.setdefault(key, []).append(i)
    weights = np.zeros(len(m))
    support_cells = []
    missing = []
    for key, mass in masses.items():
        inds = positions.get(key, [])
        if not inds:
            missing.append(dict(cell=key, benchmark_hire_mass=mass))
            continue
        sub = m.iloc[inds]
        if equal_unit:
            counts = sub[equal_unit].value_counts()
            within = 1 / sub[equal_unit].map(counts).to_numpy() / len(counts)
        else:
            within = np.repeat(1/len(inds), len(inds))
        weights[inds] = mass*within
        support_cells.append(dict(cell=key, benchmark_hire_mass=mass,
                                  monitor_rows=len(sub),
                                  monitor_candidates=sub.candidate_id.nunique()))
    supported = float(weights.sum())
    normalized = weights/supported
    cluster_w = pd.Series(normalized).groupby(m.candidate_id.reset_index(drop=True)).sum()
    nonzero = normalized[normalized > 0]
    unsupported = 1-supported
    out = dict(columns=columns, reference_specific=reference,
               within_cell_weighting=equal_unit or 'rows',
               supported_benchmark_hire_mass=supported,
               unsupported_benchmark_hire_mass=unsupported,
               row_ess=1/np.square(normalized).sum(),
               candidate_weight_ess=1/np.square(cluster_w).sum(),
               maximum_candidate_weight=float(cluster_w.max()),
               maximum_row_weight=float(normalized.max()),
               max_to_median_nonzero_row_weight=float(nonzero.max()/np.median(nonzero)),
               minimum_cell_candidates=min(c['monitor_candidates'] for c in support_cells),
               covered_cells=len(support_cells), missing_cells=missing,
               supported_cells=support_cells, arms={})
    for arm in arms:
        change = (m['margin__'+arm].to_numpy()>0).astype(float) - (m[PARENT].to_numpy()>0)
        known = float(np.dot(weights, change)*100)
        out['arms'][arm] = dict(known_contribution_pp=known,
                               supported_mass_normalized_estimate_pp=known/supported,
                               full_target_bound_pp=[known-100*unsupported, known+100*unsupported])
    return out


def cluster_bootstrap(monitor, benchmark, arms, repetitions=3000):
    """Resample monitor candidates, holding the evaluation benchmark fixed.

    The empirical selected monitor distribution is all this measures. It does
    not account for selection bias, new-candidate transfer or new-arm drift.
    """
    ids = monitor.candidate_id.astype(str).unique()
    code = pd.Categorical(monitor.candidate_id.astype(str), categories=ids).codes
    bins = monitor.bin8.to_numpy()
    C, B = len(ids), 8
    counts = np.zeros((C, B))
    np.add.at(counts, (code, bins), 1)
    n = counts.sum(axis=0)
    target = benchmark[benchmark.ref_hire]
    exact_weights = np.bincount(target.bin8, minlength=8)/len(target)
    rng = np.random.default_rng(20261022)
    multiplicities = rng.multinomial(C, np.repeat(1/C, C), size=repetitions)
    boot_n = multiplicities @ counts
    valid = (boot_n > 0).all(axis=1)
    result = dict(repetitions=repetitions, valid_repetitions=int(valid.sum()), seed=20261022,
                  cluster='monitor candidate_id', fixed_benchmark=True,
                  note='Conditional empirical-monitor uncertainty; excludes selection and transfer bias.', arms={})
    for arm in arms:
        d = (monitor['margin__'+arm].to_numpy()>0).astype(float)-(monitor[PARENT].to_numpy()>0)
        cd = np.zeros((C, B))
        np.add.at(cd, (code, bins), d)
        numer = cd.sum(axis=0)
        point = float(np.dot(ROUNDED_HIRE, numer/n)*100)
        boot_est = (multiplicities[valid] @ cd / boot_n[valid]) @ ROUNDED_HIRE * 100
        actual = float(((target['margin__'+arm]>0).astype(float)-(target[PARENT]>0)).mean()*100)
        # Deletion recomputes the means and preserves fixed benchmark weights.
        loo = ((numer-cd)/(n-counts)) @ ROUNDED_HIRE * 100
        influence = point-loo
        contributions = (cd/n) @ exact_weights * 100
        positive = np.clip(contributions, 0, None)
        negative = -np.clip(contributions, None, 0)
        top = np.argsort(np.abs(influence))[::-1][:10]
        result['arms'][arm] = dict(
            point_pp=point, actual_benchmark_pp=actual,
            projection_percentile95_pp=np.quantile(boot_est, [.025,.975]),
            optimism_percentile95_pp=np.quantile(boot_est-actual,[.025,.975]),
            empirical_fraction_projection_below_minus8=float((boot_est < -8).mean()),
            leave_one_candidate_projection_range_pp=[float(loo.min()),float(loo.max())],
            maximum_absolute_leave_one_candidate_change_pp=float(np.abs(influence).max()),
            positive_candidate_contribution_sum_pp=float(positive.sum()),
            negative_candidate_contribution_sum_pp=float(negative.sum()),
            top5_positive_candidate_share=float(np.sort(positive)[-5:].sum()/positive.sum()),
            top5_negative_candidate_share=float(np.sort(negative)[-5:].sum()/negative.sum()),
            most_influential_candidates=[dict(candidate_id=ids[i], rows=int(counts[i].sum()),
                projection_without_candidate_pp=float(loo[i]),
                point_minus_leave_one_out_pp=float(influence[i]),
                fixed_denominator_contribution_pp=float(contributions[i])) for i in top])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    df = pd.read_json(args.rows, lines=True) if '.jsonl' in args.rows.suffixes or '.ndjson' in args.rows.suffixes else pd.read_parquet(args.rows)
    arms = [c.split('margin__')[1] for c in df if c.startswith('margin__') and c != PARENT]
    for c in ['candidate_id','pair_id','protected_group','protected_attr','condition']:
        if df[c].isna().any():
            raise ValueError('Missing '+c)
        df[c] = df[c].astype(str)
    if not df.reference_decision.str.lower().isin(['hire','reject']).all():
        raise ValueError('Unexpected reference decision')
    if df.cohort.value_counts().to_dict() != {'benchmark':30600, 'monitor':696}:
        raise ValueError('Unexpected cohort sizes')
    if df.duplicated(['cohort','pair_id','protected_group','protected_attr','condition']).any():
        raise ValueError('Duplicate prompt keys within cohort')
    if set(df.condition) != {'explicit','implicit'}:
        raise ValueError('Unexpected conditions')
    if not np.isfinite(df[[PARENT,*['margin__'+a for a in arms]]].to_numpy()).all():
        raise ValueError('Nonfinite margins')
    df['ref_hire'] = df.reference_decision.str.lower() == 'hire'
    # np.digitize default puts exactly-edge values in the right-hand bin.
    df['bin8'] = np.digitize(df[PARENT], EDGES)
    b = df[df.cohort == 'benchmark'].copy()
    m = df[df.cohort == 'monitor'].copy()
    splits = b.groupby('bin8')[PARENT].median().reindex(range(8)).to_numpy()
    df['bin16'] = 2*df.bin8 + (df[PARENT].to_numpy() >= splits[df.bin8]).astype(int)
    b, m = df[df.cohort == 'benchmark'].copy(), df[df.cohort == 'monitor'].copy()
    provenance_path = args.rows.parent/'monitor_provenance.json'
    if provenance_path.exists():
        provenance = pd.DataFrame(json.loads(provenance_path.read_text()))
        for c in ['pair_id','candidate_id','protected_group','protected_attr','condition']:
            if not (provenance[c].astype(str).to_numpy() == m[c].to_numpy()).all():
                raise ValueError('Provenance order mismatch '+c)
        m['source'] = provenance.source.to_numpy()
        b['source'] = 'djinni'
    h = b[b.ref_hire]
    exact_weights = np.bincount(h.bin8, minlength=8)/len(h)
    result = dict(diagnostic_only=True, no_training=True, no_gate_or_threshold_changed=True,
                  rows_sha256=hashlib.sha256(args.rows.read_bytes()).hexdigest(),
                  rows_path=str(args.rows),
                  edges=EDGES, fixed_16bin_parent_medians=splits,
                  exact_benchmark_hire_bin_weights=exact_weights,
                  methods_note='All ablations exploratory. Bin16 cuts use benchmark parent only. No acceptance claim.',
                  arms={}, ablations={})
    if provenance_path.exists():
        result['provenance_path']=str(provenance_path)
        result['provenance_sha256']=hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    for arm in arms:
        rows=[]
        for k in range(8):
            mm,bb,hh=m[m.bin8==k],b[b.bin8==k],h[h.bin8==k]
            sm,sb,sh=summarize(mm,arm),summarize(bb,arm),summarize(hh,arm)
            w=exact_weights[k]
            rows.append(dict(bin=k, benchmark_hire_weight=w, monitor_all=sm,
                benchmark_all=sb, benchmark_ref_hire=sh,
                monitor_ref_hire=summarize(mm[mm.ref_hire],arm) if mm.ref_hire.any() else None,
                weighted_monitor_loss_pp=-w*sm['loss_pp'],
                weighted_monitor_gain_pp=w*sm['gain_pp'],
                weighted_benchmark_hire_loss_pp=-w*sh['loss_pp'],
                weighted_benchmark_hire_gain_pp=w*sh['gain_pp'],
                weighted_total_error_pp=w*(sm['delta_pp']-sh['delta_pp']),
                weighted_population_transfer_error_pp=w*(sm['delta_pp']-sb['delta_pp']),
                weighted_benchmark_label_pooling_error_pp=w*(sb['delta_pp']-sh['delta_pp'])))
        rawmonitor=sum(r['weighted_monitor_loss_pp']+r['weighted_monitor_gain_pp'] for r in rows)
        actual=summarize(h,arm)['delta_pp']
        prod=sum(ROUNDED_HIRE[r['bin']]*r['monitor_all']['delta_pp'] for r in rows)
        assert abs(sum(r['weighted_total_error_pp'] for r in rows)-(rawmonitor-actual)) < 1e-10
        result['arms'][arm]=dict(
            bins=rows, production_projection_pp=prod,
            benchmark_actual_pp=actual,
            exact_projection_pp=rawmonitor,
            projection_loss_pp=sum(r['weighted_monitor_loss_pp'] for r in rows),
            projection_gain_pp=sum(r['weighted_monitor_gain_pp'] for r in rows),
            benchmark_loss_pp=sum(r['weighted_benchmark_hire_loss_pp'] for r in rows),
            benchmark_gain_pp=sum(r['weighted_benchmark_hire_gain_pp'] for r in rows),
            population_transfer_error_pp=sum(r['weighted_population_transfer_error_pp'] for r in rows),
            benchmark_label_pooling_error_pp=sum(r['weighted_benchmark_label_pooling_error_pp'] for r in rows),
            weight_rounding_pp=prod-rawmonitor)
    fixed_ablations = [
        ('bin8', ['bin8'],False,None),
        ('bin16', ['bin16'],False,None),
        ('bin8_condition', ['bin8','condition'],False,None),
        ('bin8_group_condition', ['bin8','protected_group','condition'],False,None),
        ('bin8_attribute_condition', ['bin8','protected_attr','protected_group','condition'],False,None),
        ('bin8_reference', ['bin8'],True,None),
        ('bin16_reference', ['bin16'],True,None),
        ('bin8_condition_reference', ['bin8','condition'],True,None),
        ('bin8_group_condition_reference', ['bin8','protected_group','condition'],True,None),
        ('bin8_attribute_condition_reference', ['bin8','protected_attr','protected_group','condition'],True,None),
        ('bin8_equal_candidate', ['bin8'],False,'candidate_id'),
        ('bin8_equal_pair', ['bin8'],False,'pair_id'),
    ]
    for name,cols,ref,unit in fixed_ablations:
        result['ablations'][name]=estimator(m,h,arms,cols,ref,unit)
    if 'source' in m:
        h = h.assign(source='djinni')
        for name,cols,ref in [
            ('bin8_source',['bin8','source'],False),
            ('bin8_source_group_condition',['bin8','source','protected_group','condition'],False),
            ('bin8_source_reference',['bin8','source'],True),
        ]:
            result['ablations'][name]=estimator(m,h,arms,cols,ref)
        result['source_bin_summaries']={source:{arm:[dict(bin=k,**summarize(g,arm)) for k,g in sub.groupby('bin8')] for arm in MAIN} for source,sub in m.groupby('source')}
        result['source_cluster_bootstrap']={source:cluster_bootstrap(sub,b,MAIN) for source,sub in m.groupby('source')}
        result['source_mixture_decomposition']={}
        for arm in arms:
            terms=[]
            for k in range(8):
                mm=m[m.bin8==k]
                dj=mm[mm.source=='djinni']
                rj=mm[mm.source=='rjdb']
                hh=h[h.bin8==k]
                dm=summarize(dj,arm)['delta_pp']
                rm=summarize(rj,arm)['delta_pp']
                bm=summarize(hh,arm)['delta_pp']
                terms.append(dict(bin=k, djinni_rows=len(dj), rjdb_rows=len(rj),
                    djinni_candidates=dj.candidate_id.nunique(),
                    rjdb_candidates=rj.candidate_id.nunique(),
                    monitor_rjdb_share=len(rj)/len(mm),
                    djinni_change_pp=dm, rjdb_change_pp=rm,
                    benchmark_ref_hire_change_pp=bm,
                    source_mixture_effect_pp=exact_weights[k]*len(rj)/len(mm)*(rm-dm),
                    djinni_residual_error_pp=exact_weights[k]*(dm-bm)))
            result['source_mixture_decomposition'][arm]=dict(bins=terms,
                source_mixture_effect_pp=sum(t['source_mixture_effect_pp'] for t in terms),
                djinni_residual_error_pp=sum(t['djinni_residual_error_pp'] for t in terms),
                note='Algebraic decomposition, not an identified causal source effect.')
            sd=result['source_mixture_decomposition'][arm]
            formal=result['arms'][arm]
            assert abs(sd['source_mixture_effect_pp']+sd['djinni_residual_error_pp']+
                       formal['weight_rounding_pp']-
                       (formal['production_projection_pp']-formal['benchmark_actual_pp'])) < 1e-10
    result['cluster_bootstrap']=cluster_bootstrap(m,b,MAIN)
    result['cohorts']={}
    for name,x in [('monitor',m),('benchmark',b)]:
        result['cohorts'][name]=dict(rows=len(x), candidates=x.candidate_id.nunique(),
            pairs=x.pair_id.nunique(), reference_hire_rows=int(x.ref_hire.sum()),
            by_bin=[dict(bin=k,rows=len(g),candidates=g.candidate_id.nunique(),
                pairs=g.pair_id.nunique(),reference_hire_rows=int(g.ref_hire.sum()),
                reference_hire_candidates=g[g.ref_hire].candidate_id.nunique()) for k,g in x.groupby('bin8')],
            condition_counts=x.condition.value_counts().to_dict(),
            protected_group_counts=x.protected_group.value_counts().to_dict(),
            attr_group_counts=x.groupby(['protected_attr','protected_group']).size().to_dict())
    args.output.write_text(json.dumps(jsonable(result),indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({a:{k:v for k,v in r.items() if k!='bins'} for a,r in result['arms'].items()},indent=2))


if __name__ == '__main__':
    main()
