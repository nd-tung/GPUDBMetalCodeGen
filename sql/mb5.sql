-- MB5: nearly-non-selective filter (~99%)
-- Isolates: filter overhead when nearly every row passes.
-- Compare to MB2 (no filter) to measure pure filter-evaluation cost.

SELECT
    SUM(l_extendedprice) AS revenue
FROM
    lineitem
WHERE
    l_shipdate >= DATE '1992-01-01';
