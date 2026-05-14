-- S28: IN subquery (ANY_SUBLINK)
SELECT o_orderkey, o_orderdate
FROM orders
WHERE o_orderdate >= DATE '1996-01-01'
  AND o_orderdate <  DATE '1996-01-03'
  AND o_orderkey IN (
      SELECT l_orderkey FROM lineitem WHERE l_linenumber = 1
  )
LIMIT 10
