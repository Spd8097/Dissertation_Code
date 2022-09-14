-- table 1, base table
DROP TABLE IF EXISTS ${schema}.parameters_for_analysis_tmp;
CREATE TABLE ${schema}.parameters_for_analysis_tmp AS
SELECT a.sku_key,
       a.quantity,
       a.plu_amt,
       a.tran_discount_amt,
       a.override_amt,
       a.extended_amt,
       CASE
           WHEN a.ecom = 'Y' THEN 1
           WHEN a.ecom = 'N' THEN 0
           END                                  AS ecom,
       CASE
           WHEN a.sale = 'Y' THEN 1
           WHEN a.sale = 'N' THEN 0
           END                                  AS sale,
       CASE
           WHEN a.return = 'Y' THEN 1
           WHEN a.return = 'N' THEN 0
           END                                  AS return,
       CASE
           WHEN a.receipted = 'Y' THEN 1
           WHEN a.receipted = 'N' THEN 0
           END                                  AS receipted,
       b.class_desc || ' ' || b.department_desc AS product_desc,
       b.sku_cost,
       split_part(b.sku_desc, '/', 2)           AS sku_color,
       split_part(b.sku_desc, '/', 3)           AS sku_size
FROM ardm_095.std_clnt_detail a
         INNER JOIN ardm_095.std_clnt_sku b ON a.sku_key = b.sku_key
WHERE a.date_pos >= '2021-01-01'
  AND a.date_pos < '2022-01-01';


-- table 2, final table
DROP TABLE IF EXISTS ${schema}.parameters_for_analysis;
CREATE TABLE ${schema}.parameters_for_analysis_cln AS
SELECT sku_key,                                                           -- identifier
       AVG(return)                                  AS return_rate,       -- outcome variable
       product_desc,                                                      -- item description
       sku_color,                                                         -- item color
       sku_size,                                                          -- item size
       sku_cost,                                                          -- item cost
       AVG(CASE WHEN sale = 1 THEN plu_amt END)     AS avg_price_product, -- avg sale price of item
       SUM(CASE WHEN sale = 1 THEN quantity END)    AS quantity,          -- total sales of item
       AVG(CASE WHEN sale = 1 THEN ecom END)        AS ecom_rate,         -- proportion of item sales that are ecom
       AVG(CASE WHEN return = 1 THEN receipted END) AS receipted_rate     -- proportion of item returns that are receipted
FROM ${schema}.parameters_for_analysis_tmp
WHERE sku_key IN (SELECT sku_key
                  FROM ${schema}.parameters_for_analysis_tmp
                  GROUP BY sku_key
                  HAVING SUM(sale) > 100) -- only use items that have been sold more than 100 times
GROUP BY sku_key, product_desc, sku_color, sku_size, sku_cost;

select * from parameters_for_analysis_cln
