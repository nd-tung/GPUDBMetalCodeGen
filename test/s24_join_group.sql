-- S24: Join + GROUP BY
SELECT o_orderpriority, COUNT(*) AS cnt
FROM lineitem, orders
WHERE l_orderkey = o_orderkey
  AND o_orderdate >= DATE '1996-01-01'
  AND o_orderdate <  DATE '1996-02-01'
  AND o_orderpriority IN ('1-URGENT', '2-HIGH')
GROUP BY o_orderpriority
ORDER BY o_orderpriority
