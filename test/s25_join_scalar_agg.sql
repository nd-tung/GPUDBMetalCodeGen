-- S25: Join + scalar aggregate
SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM lineitem, orders
WHERE l_orderkey = o_orderkey
  AND o_orderdate >= DATE '1996-01-01'
  AND o_orderdate <  DATE '1996-02-01'
  AND o_orderpriority = '1-URGENT'
