#!/usr/bin/env python3
"""Analyze SF100 chunk size / double-buffer experiment results."""
import csv, io, statistics
from collections import defaultdict

data = """query,chunk_M,double_buffer,rep,chunks,parse_ms,gpu_ms
q1,5,yes,1,115,199162.0,3698.1
q1,5,yes,2,115,175940.4,3740.3
q1,5,yes,3,115,187485.2,3763.6
q6,5,yes,1,115,122629.1,21.6
q6,5,yes,2,115,122595.2,23.1
q6,5,yes,3,115,122346.5,21.2
q13,5,yes,1,29,35823.5,69.6
q13,5,yes,2,29,12978.7,69.3
q13,5,yes,3,29,12942.4,70.6
q17,5,yes,1,115,70740.6,680.9
q17,5,yes,2,115,70364.1,669.8
q17,5,yes,3,115,70235.7,839.7
q1,5,no,1,115,195004.6,3734.7
q1,5,no,2,115,199294.9,3640.9
q1,5,no,3,115,176301.3,3767.1
q6,5,no,1,115,122249.1,24.5
q6,5,no,2,115,106776.7,24.1
q6,5,no,3,115,95561.4,23.4
q13,5,no,1,29,12929.0,73.3
q13,5,no,2,29,39904.7,75.0
q13,5,no,3,29,12949.7,68.1
q17,5,no,1,115,70304.2,837.3
q17,5,no,2,115,70837.3,892.5
q17,5,no,3,115,64922.1,868.7
q1,10,yes,1,58,179183.0,3290.2
q1,10,yes,2,58,202782.0,3302.5
q1,10,yes,3,58,197373.1,3333.8
q6,10,yes,1,58,124837.3,10.5
q6,10,yes,2,58,82374.2,10.6
q6,10,yes,3,58,88623.5,10.5
q13,10,yes,1,15,12982.7,67.0
q13,10,yes,2,15,13007.1,67.6
q13,10,yes,3,15,33770.9,67.8
q17,10,yes,1,58,56719.0,814.2
q17,10,yes,2,58,70061.1,787.6
q17,10,yes,3,58,70238.6,739.4
q1,10,no,1,58,198494.9,3314.8
q1,10,no,2,58,185485.1,3320.4
q1,10,no,3,58,170942.0,3362.3
q6,10,no,1,58,117419.9,10.6
q6,10,no,2,58,122394.5,10.8
q6,10,no,3,58,76412.8,10.7
q13,10,no,1,15,19368.1,70.4
q13,10,no,2,15,12958.5,68.3
q13,10,no,3,15,12959.0,63.5
q17,10,no,1,58,67564.0,672.3
q17,10,no,2,58,70044.8,839.8
q17,10,no,3,58,47315.8,683.7
q1,25,yes,1,23,192316.7,3177.6
q1,25,yes,2,23,198953.2,3128.0
q1,25,yes,3,23,163235.3,3182.8
q6,25,yes,1,23,96381.4,9.5
q6,25,yes,2,23,122137.0,9.3
q6,25,yes,3,23,99188.5,9.4
q13,25,yes,1,6,13165.1,65.5
q13,25,yes,2,6,13066.3,64.9
q13,25,yes,3,6,36101.7,55.7
q17,25,yes,1,23,70189.5,540.9
q17,25,yes,2,23,70222.4,646.1
q17,25,yes,3,23,169505.6,648.8
q1,25,no,1,23,172549.3,3192.4
q1,25,no,2,23,171253.5,3150.4
q1,25,no,3,23,196940.0,3193.2
q6,25,no,1,23,99342.8,9.7
q6,25,no,2,23,122470.8,9.7
q6,25,no,3,23,99485.4,9.7
q13,25,no,1,6,12958.4,67.6
q13,25,no,2,6,12980.3,67.9
q13,25,no,3,6,35919.3,63.5
q17,25,no,1,23,90626.6,649.8
q17,25,no,2,23,108285.2,615.3
q17,25,no,3,23,52653.9,527.2
q1,50,yes,1,12,196192.0,3111.2
q1,50,yes,2,12,198342.4,3097.2
q1,50,yes,3,12,177637.3,3108.1
q6,50,yes,1,12,99389.2,8.3
q6,50,yes,2,12,121995.5,8.2
q6,50,yes,3,12,108986.5,8.4
q13,50,yes,1,3,19027.8,59.5
q13,50,yes,2,3,13256.2,65.7
q13,50,yes,3,3,13257.7,50.8
q17,50,yes,1,12,52837.9,499.2
q17,50,yes,2,12,118965.1,605.1
q17,50,yes,3,12,67156.4,510.7
q1,50,no,1,12,155537.7,3087.3
q1,50,no,2,12,183775.8,3094.1
q1,50,no,3,12,194998.4,3105.1
q6,50,no,1,12,102971.9,8.6
q6,50,no,2,12,99205.3,8.5
q6,50,no,3,12,144211.3,9.2
q13,50,no,1,3,34084.6,54.3
q13,50,no,2,3,13100.1,45.3
q13,50,no,3,3,13070.6,59.4
q17,50,no,1,12,69932.1,504.5
q17,50,no,2,12,69379.0,559.4
q17,50,no,3,12,52902.6,556.5
q1,100,yes,1,6,199344.3,3063.6
q1,100,yes,2,6,189856.4,3067.7
q1,100,yes,3,6,176362.2,3063.5
q6,100,yes,1,6,147607.1,7.7
q6,100,yes,2,6,122355.6,7.8
q6,100,yes,3,6,83548.6,7.6
q13,100,yes,1,2,13485.8,46.9
q13,100,yes,2,2,13388.4,50.4
q13,100,yes,3,2,33019.3,48.8
q17,100,yes,1,6,70563.2,513.5
q17,100,yes,2,6,67579.4,463.1
q17,100,yes,3,6,131391.5,466.0
q1,100,no,1,6,192759.0,3041.0
q1,100,no,2,6,152702.1,3219.0
q1,100,no,3,6,195442.7,3067.7
q6,100,no,1,6,145973.7,8.3
q6,100,no,2,6,82533.9,7.7
q6,100,no,3,6,101930.9,7.8
q13,100,no,1,2,13343.5,46.3
q13,100,no,2,2,19955.6,46.7
q13,100,no,3,2,13228.2,53.0
q17,100,no,1,6,109554.4,462.6
q17,100,no,2,6,54430.7,523.0
q17,100,no,3,6,70333.2,548.3
q1,200,yes,1,3,201858.1,3045.4
q1,200,yes,2,3,240104.3,3021.9
q1,200,yes,3,3,163790.1,3040.6
q6,200,yes,1,3,122375.8,7.5
q6,200,yes,2,3,161459.7,7.5
q6,200,yes,3,3,99978.6,7.5
q13,200,yes,1,1,13474.1,44.3
q13,200,yes,2,1,13587.5,44.0
q13,200,yes,3,1,37984.0,42.5
q17,200,yes,1,3,58835.5,458.2
q17,200,yes,2,3,57060.3,442.5
q17,200,yes,3,3,90468.8,469.0
q1,200,no,1,3,198717.8,3045.3
q1,200,no,2,3,173550.8,3040.7
q1,200,no,3,3,202310.4,3035.4
q6,200,no,1,3,99078.7,7.7
q6,200,no,2,3,128117.0,7.5
q6,200,no,3,3,126368.7,7.6
q13,200,no,1,1,36514.4,42.7
q13,200,no,2,1,13506.9,45.6
q13,200,no,3,1,13574.1,43.9
q17,200,no,1,3,66165.9,437.7
q17,200,no,2,3,136505.9,451.3
q17,200,no,3,3,134725.7,438.7"""

reader = csv.DictReader(io.StringIO(data))
rows = list(reader)

groups = defaultdict(list)
for r in rows:
    key = (r['query'], int(r['chunk_M']), r['double_buffer'])
    groups[key].append({
        'gpu': float(r['gpu_ms']),
        'parse': float(r['parse_ms']),
        'chunks': int(r['chunks'])
    })

def med(vals): return statistics.median(vals)

# ========= GPU TIME TABLE =========
print("=" * 85)
print("TABLE 1: GPU TIME (ms) — Median of 3 runs")
print("=" * 85)
print(f"{'Query':<6} {'Chunk':<8} {'Chunks':>7} {'DB=yes':>10} {'DB=no':>10} {'DB->noDB':>10}")
print("-" * 85)

for q in ['q1', 'q6', 'q13', 'q17']:
    for c in [5, 10, 25, 50, 100, 200]:
        g_yes = groups[(q, c, 'yes')]
        g_no  = groups[(q, c, 'no')]
        m_yes = med([x['gpu'] for x in g_yes])
        m_no  = med([x['gpu'] for x in g_no])
        chunks = g_yes[0]['chunks']
        diff = ((m_no - m_yes) / m_no * 100)
        print(f"{q:<6} {str(c)+'M':<8} {chunks:>7} {m_yes:>10.1f} {m_no:>10.1f} {diff:>+9.1f}%")
    print()

# ========= PARSE TIME TABLE =========
print("=" * 85)
print("TABLE 2: CPU PARSE TIME (ms) — Median of 3 runs")
print("=" * 85)
print(f"{'Query':<6} {'Chunk':<8} {'DB=yes':>12} {'DB=no':>12} {'Diff':>10}")
print("-" * 85)

for q in ['q1', 'q6', 'q13', 'q17']:
    for c in [5, 10, 25, 50, 100, 200]:
        g_yes = groups[(q, c, 'yes')]
        g_no  = groups[(q, c, 'no')]
        m_yes = med([x['parse'] for x in g_yes])
        m_no  = med([x['parse'] for x in g_no])
        diff = ((m_no - m_yes) / m_no * 100)
        print(f"{q:<6} {str(c)+'M':<8} {m_yes:>12.0f} {m_no:>12.0f} {diff:>+9.1f}%")
    print()

# ========= WALL CLOCK TABLE =========
print("=" * 85)
print("TABLE 3: TOTAL WALL TIME parse+GPU (ms) — Median of 3 runs")
print("=" * 85)
print(f"{'Query':<6} {'Chunk':<8} {'DB=yes':>12} {'DB=no':>12} {'DB benefit':>12}")
print("-" * 85)

for q in ['q1', 'q6', 'q13', 'q17']:
    for c in [5, 10, 25, 50, 100, 200]:
        g_yes = groups[(q, c, 'yes')]
        g_no  = groups[(q, c, 'no')]
        w_yes = med([x['parse'] + x['gpu'] for x in g_yes])
        w_no  = med([x['parse'] + x['gpu'] for x in g_no])
        diff = ((w_no - w_yes) / w_no * 100)
        print(f"{q:<6} {str(c)+'M':<8} {w_yes:>12.0f} {w_no:>12.0f} {diff:>+10.1f}%")
    print()

# ========= BEST CONFIGS =========
print("=" * 85)
print("BEST CONFIG per query (lowest median GPU ms)")
print("=" * 85)
for q in ['q1', 'q6', 'q13', 'q17']:
    best_key, best_gpu = None, float('inf')
    for (qk, c, db), vals in groups.items():
        if qk != q: continue
        m = med([x['gpu'] for x in vals])
        if m < best_gpu:
            best_gpu = m
            best_key = (c, db)
    c, db = best_key
    chunks = groups[(q, c, db)][0]['chunks']
    print(f"  {q}: chunk={c}M db={db} -> {best_gpu:.1f}ms ({chunks} chunks)")

# ========= GPU SCALING =========
print()
print("=" * 85)
print("GPU SCALING: Speedup vs 5M baseline (db=yes)")
print("=" * 85)
for q in ['q1', 'q6', 'q13', 'q17']:
    base = med([x['gpu'] for x in groups[(q, 5, 'yes')]])
    parts = []
    for c in [5, 10, 25, 50, 100, 200]:
        m = med([x['gpu'] for x in groups[(q, c, 'yes')]])
        parts.append(f"{c}M={m:.0f}({base/m:.2f}x)")
    print(f"  {q}: " + " | ".join(parts))
