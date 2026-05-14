-- S11: CASE WHEN expression (numeric result)
SELECT 
    l_orderkey,
    l_quantity,
    CASE 
        WHEN l_quantity < 20 THEN l_quantity * 2
        WHEN l_quantity < 40 THEN l_quantity * 3
        ELSE l_quantity
    END AS scaled_qty
FROM lineitem
WHERE l_orderkey BETWEEN 1 AND 10
