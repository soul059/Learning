---
tags:
  - sql
  - aggregate-functions
  - group-by
  - having
  - dql
---

# SQL Aggregate Functions and Grouping

[[SQL|Aggregate functions]] perform calculations on a set of values and return a single value. They are frequently used with the [[SQL DQL|SELECT]] statement to summarize data. The `GROUP BY` clause allows you to apply these aggregate functions to subsets of rows based on common values in one or more columns.

## SQL Aggregate Functions

These functions operate on a collection of values from a column or expression.

* **COUNT()**: Returns the number of rows that match a specified criterion.
    * `COUNT(*)`: Counts all rows in the result set (or group).
    * `COUNT(column_name)`: Counts rows where `column_name` is *not* [[SQL NULL Values|NULL]].
    * `COUNT(DISTINCT column_name)`: Counts the number of distinct non-null values in `column_name`.
* **SUM()**: Returns the total sum of a numeric column. Ignores [[SQL NULL Values|NULL]] values.
* **AVG()**: Returns the average value of a numeric column. Ignores [[SQL NULL Values|NULL]] values.
* **MIN()**: Returns the minimum value in a column. Ignores [[SQL NULL Values|NULL]] values.
* **MAX()**: Returns the maximum value in a column. Ignores [[SQL NULL Values|NULL]] values.

**Examples (using a hypothetical `Orders` table with columns `OrderID`, `CustomerID`, `OrderDate`, `Amount`):**

```sql
-- Get the total number of orders
SELECT COUNT(*) FROM Orders;
```

```sql
-- Get the number of orders with a non-null CustomerID
SELECT COUNT(CustomerID) FROM Orders;
```

```sql
-- Get the number of unique customers who placed orders
SELECT COUNT(DISTINCT CustomerID) FROM Orders;
```

```sql
-- Get the total amount of all orders
SELECT SUM(Amount) FROM Orders;
```

```sql
-- Get the average order amount
SELECT AVG(Amount) FROM Orders;
```

```sql
-- Get the smallest order amount
SELECT MIN(Amount) FROM Orders;
```

```sql
-- Get the largest order amount
SELECT MAX(Amount) FROM Orders;
```

## GROUP BY Clause

The `GROUP BY` clause is used with [[SQL Aggregate Functions and Grouping|aggregate functions]] to group the result-set by one or more columns. The aggregate function then calculates a summary value for each group.

```sql
-- Basic syntax
SELECT column1, column2, aggregate_function(column3)
FROM table_name
WHERE condition
GROUP BY column1, column2;
```
* Any column in the `SELECT` list that is *not* an aggregate function must be included in the `GROUP BY` clause.

**Example (using `Orders` table):**

```sql
-- Find the total amount spent by each customer
SELECT CustomerID, SUM(Amount) AS TotalSpent
FROM Orders
GROUP BY CustomerID;
```

```sql
-- Find the number of orders and average amount per customer per day
SELECT CustomerID, OrderDate, COUNT(OrderID) AS NumberOfOrders, AVG(Amount) AS AverageOrderAmount
FROM Orders
GROUP BY CustomerID, OrderDate -- Group by both columns
ORDER BY CustomerID, OrderDate; -- Order the result for readability
```

## HAVING Clause

The `HAVING` clause is used to filter groups created by the `GROUP BY` clause. It is similar to the `WHERE` clause, but `WHERE` filters *rows before* grouping, while `HAVING` filters *groups after* grouping and aggregation.

```sql
-- Syntax
SELECT column1, aggregate_function(column2)
FROM table_name
WHERE row_condition -- Optional: filter rows before grouping
GROUP BY column1
HAVING group_condition; -- Filter groups based on aggregate results or grouping columns
```

**Example (using `Orders` table):**

```sql
-- Find customers who have placed more than 5 orders
SELECT CustomerID, COUNT(OrderID) AS NumberOfOrders
FROM Orders
GROUP BY CustomerID
HAVING COUNT(OrderID) > 5; -- Filter groups where count is > 5
```

```sql
-- Find customers whose total spending is over 1000
SELECT CustomerID, SUM(Amount) AS TotalSpent
FROM Orders
GROUP BY CustomerID
HAVING SUM(Amount) > 1000;
```

```sql
-- Find customers who placed more than 3 orders on a single day
SELECT CustomerID, OrderDate, COUNT(OrderID) AS OrdersOnDay
FROM Orders
GROUP BY CustomerID, OrderDate
HAVING COUNT(OrderID) > 3;
```
The `HAVING` clause is essential for filtering based on conditions that involve aggregate results.

Aggregate functions and grouping are powerful tools for summarizing and analyzing data in your database.

---
**Related Notes:**
* [[SQL]]
* [[SQL DQL]]
* [[Relational Algebra]] (links to aggregate function concept)
* [[SQL NULL Values]]