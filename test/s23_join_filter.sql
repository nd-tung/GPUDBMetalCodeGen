-- S23: Join with WHERE filter
SELECT o_orderkey, l_quantity, o_orderdate
FROM lineitem, orders
WHERE l_orderkey = o_orderkey
  AND o_orderdate >= DATE '1996-01-01'
  AND o_orderdate <  DATE '1996-01-07'
  AND l_linenumber = 1
ORDER BY o_orderkey
LIMIT 10
