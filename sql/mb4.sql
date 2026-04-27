-- MB4: highly selective filter (~1%)
-- Isolates: behavior under sparse selection (most threads short-circuit).
-- Compare to MB3 (50%) and MB5 (~99%) to plot selectivity-vs-throughput.

SELECT
    SUM(l_extendedprice) AS revenue
FROM
    lineitem
WHERE
    l_shipdate >= DATE '1998-09-01'
    AND l_shipdate < DATE '1998-10-01';
