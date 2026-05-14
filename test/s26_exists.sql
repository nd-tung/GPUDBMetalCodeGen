-- S26: EXISTS subquery
SELECT o_orderkey, o_orderdate
FROM orders
WHERE o_orderdate >= DATE '1996-01-01'
  AND o_orderdate <  DATE '1996-01-07'
  AND EXISTS (
      SELECT * FROM lineitem
      WHERE l_orderkey = o_orderkey
        AND l_linenumber = 1
  )
LIMIT 10
