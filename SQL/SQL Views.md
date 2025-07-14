---
tags:
  - sql
  - views
  - virtual-table
  - ddl
  - dql
---

# SQL Views

[[SQL|A view]] is a virtual table based on the result-set of a [[SQL DQL|SELECT]] statement. A view contains rows and columns, just like a real table, but it does not store data physically. The data accessed through a view is stored in the underlying base table(s).

Think of a view as a stored query that you can reference as if it were a table.

## Creating a View

Views are created using the [[SQL DDL|CREATE VIEW]] statement.

```sql
CREATE VIEW view_name AS
SELECT column1, column2, ...
FROM table_name
WHERE condition;
```

**Example:**

```sql
-- Create a view showing only students from the 'Computer Science' major
CREATE VIEW CS_Students AS
SELECT StudentID, FirstName, LastName, EnrollmentDate
FROM Students
WHERE MajorID = (SELECT MajorID FROM Majors WHERE MajorName = 'Computer Science');
```

```sql
-- Create a view showing customer order summaries
CREATE VIEW CustomerOrderSummary AS
SELECT CustomerID, COUNT(OrderID) AS NumberOfOrders, SUM(Amount) AS TotalAmount
FROM Orders
GROUP BY CustomerID;
```

## Using a View

Once created, you can query a view just like you would query a table using `SELECT`.

```sql
-- Query the CS_Students view
SELECT FirstName, LastName FROM CS_Students WHERE EnrollmentDate %3E= '2024-01-01';
```

```sql
-- Query the CustomerOrderSummary view
SELECT * FROM CustomerOrderSummary WHERE TotalAmount > 1000;
```
When you query a view, the database system executes the underlying `SELECT` statement defined in the view definition and returns the result.

## Advantages of Using Views

* **Security:** You can restrict user access to specific data by granting permissions only on the view, not the underlying tables. This hides sensitive data (columns or rows).
* **Simplification:** Complex queries (involving joins, aggregations, subqueries) can be predefined as views. Users can then query the view with simple `SELECT` statements.
* **Consistency:** Provides a consistent view of data for different applications or users.
* **Abstraction:** Hides the complexity of the underlying database structure. If the base table schema changes (e.g., adding a column), the view definition might need updating, but applications querying the view may not need to change as long as the view's columns remain the same.

## Updating Data Through Views

Modifying data (using `INSERT`, `UPDATE`, `DELETE`) through a view is possible but subject to significant restrictions.

* **Limitations:** Updates through views are generally only possible for **simple views** that meet criteria such as:
    * Based on a **single base table**.
    * Do **not** contain [[SQL Aggregate Functions and Grouping|aggregate functions]], `GROUP BY`, `HAVING`, `DISTINCT`.
    * Do **not** contain [[SQL Joins|joins]] (in most cases).
    * Include all [[SQL Constraints|NOT NULL]] columns from the base table that don't have default values.
* If an update *is* possible, the DBMS translates the modification request on the view into a corresponding modification on the underlying base table(s).
* Many complex views are **read-only**.

## Dropping a View

Views are deleted using the [[SQL DDL|DROP VIEW]] statement.

```sql
DROP VIEW view_name;
```

Views are powerful tools for managing complexity and security in relational databases, acting as customizable windows into your data.

---
**Related Notes:**
* [[SQL]]
* [[SQL DDL]]
* [[SQL DQL]]
* [[Database Concepts]] (linking to Views in general)
* [[Relational Model]]