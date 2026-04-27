-- MB6: arithmetic-heavy aggregate (4 columns, 3 multiplies, 1 subtract)
-- Isolates: ALU vs memory bandwidth balance with no filter.
-- Compare to MB2 (1 column) to attribute cost of extra column reads + arithmetic.

SELECT
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS charge
FROM
    lineitem;
