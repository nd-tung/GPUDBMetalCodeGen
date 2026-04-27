-- MB3: filter selectivity ~50% (date in mid-range)
-- Isolates: branch divergence cost on a moderately selective filter
-- with one filter column read + one aggregate column read.

SELECT
    SUM(l_extendedprice) AS revenue
FROM
    lineitem
WHERE
    l_shipdate >= DATE '1994-01-01'
    AND l_shipdate < DATE '1996-01-01';
