-- S31: Date + interval arithmetic
SELECT o_orderkey, o_orderdate
FROM orders
WHERE o_orderdate >= DATE '1995-01-01' 
  AND o_orderdate <  DATE '1995-01-01' + INTERVAL '3' MONTH
LIMIT 10
