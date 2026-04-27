-- MB7: many-column scan (5 columns, no arithmetic)
-- Isolates: pure columnar bandwidth scaling with column count.
-- Compare to MB2 (1 col) and MB6 (4 col + arith). Same row count, different column footprint.

SELECT
    SUM(l_quantity + l_extendedprice + l_discount + l_tax) AS s
FROM
    lineitem
WHERE
    l_shipdate >= DATE '1992-01-01';
