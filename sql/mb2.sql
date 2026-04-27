-- MB2: single-column scan + sum (no filter)
-- Isolates: 1 column read + reduction. Memory-bandwidth bound on lineitem.
-- Compare to MB1 to attribute cost of column read vs dispatch.

SELECT
    SUM(l_extendedprice) AS total
FROM
    lineitem;
