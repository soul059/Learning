---
tags:
  - sql
  - constraints
  - data-integrity
  - primary-key
  - unique
  - not-null
---

# SQL Constraints

[[SQL|SQL Constraints]] are rules enforced on columns or tables in a database to limit the type of data that can go into a table. This ensures the accuracy, reliability, and integrity of the data. Constraints are a key part of [[SQL DDL|data definition]] and [[Database Design|database design]].

Constraints can be defined when the table is created (`CREATE TABLE`) or after the table is created (`ALTER TABLE`).

## Common SQL Constraints

* **NOT NULL:** Ensures that a column cannot contain [[SQL NULL Values|NULL]] values. Every row must have a value for this column.

    ```sql
    CREATE TABLE table_name (
        column1 datatype NOT NULL,
        column2 datatype,
        ...
    );
    ```
    ```sql
    ALTER TABLE table_name
    ALTER COLUMN column1 datatype NOT NULL; -- Syntax varies (ALTER COLUMN or MODIFY COLUMN)
    ```

* **UNIQUE:** Ensures that all values in a column (or a set of columns) are distinct. While a [[Primary Key]] is automatically unique, you can apply `UNIQUE` to other columns.

    ```sql
    CREATE TABLE table_name (
        column1 datatype UNIQUE,
        column2 datatype,
        ...
    );
    ```
    ```sql
    -- Or as a table-level constraint (for multiple columns)
    CREATE TABLE table_name (
        column1 datatype,
        column2 datatype,
        CONSTRAINT UQ_column1_column2 UNIQUE (column1, column2),
        ...
    );
    ```
    ```sql
    ALTER TABLE table_name
    ADD CONSTRAINT UQ_column1 UNIQUE (column1);
    ```

* **PRIMARY KEY:** Uniquely identifies each row in a table.
    * A table can have only **one** primary key.
    * A primary key can consist of one or more columns.
    * Primary key values must be **unique** and **not null**.
    * Corresponds to the [[Primary Key]] concept in the [[Relational Model]]. Enforces [[Entity Integrity Constraint]].

    ```sql
    CREATE TABLE table_name (
        column1 datatype,
        column2 datatype,
        CONSTRAINT PK_column1 PRIMARY KEY (column1), -- Column-level
        ...
    );
    ```
    ```sql
    -- Or as a table-level constraint (required for composite primary keys)
    CREATE TABLE table_name (
        column1 datatype,
        column2 datatype,
        CONSTRAINT PK_column1_column2 PRIMARY KEY (column1, column2),
        ...
    );
    ```
    ```sql
    ALTER TABLE table_name
    ADD CONSTRAINT PK_column1 PRIMARY KEY (column1);
    ```

* **FOREIGN KEY:** A set of attributes in one table (the child or referencing table) that refers to the [[Primary Key]] of another table (the parent or referenced table).
    * Establishes and enforces a link between data in two tables.
    * Ensures [[Referential Integrity Constraint|referential integrity]].
    * Corresponds to the [[Foreign Key]] concept in the [[Relational Model]].

    ```sql
    CREATE TABLE Orders (
        OrderID INT PRIMARY KEY,
        CustomerID INT,
        OrderDate DATE,
        FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID) -- References the Customers table's PK
        ...
    );
    ```
    ```sql
    ALTER TABLE Orders
    ADD CONSTRAINT FK_Orders_Customers
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID);
    ```
    *Foreign key constraints can also define `ON DELETE` and `ON UPDATE` actions (e.g., `CASCADE`, `SET NULL`, `RESTRICT`) to specify behavior when the referenced primary key is changed or deleted.*

* **CHECK:** Ensures that all values in a column satisfy a specific condition.

    ```sql
    CREATE TABLE table_name (
        column1 datatype,
        column2 datatype,
        CONSTRAINT CK_column1_positive CHECK (column1 %3E= 0),
        ...
    );
    ```
    ```sql
    -- Or checking a condition across multiple columns
    CREATE TABLE Products (
        ProductID INT PRIMARY KEY,
        Price DECIMAL(10,2),
        Discount DECIMAL(10,2),
        CONSTRAINT CK_DiscountPrice CHECK (Discount %3C= Price),
        ...
    );
    ```
    ```sql
    ALTER TABLE table_name
    ADD CONSTRAINT CK_column1 CHECK (column1 IN ('A', 'B', 'C'));
    ```

* **DEFAULT:** Provides a default value for a column when no value is explicitly specified during an `INSERT` operation.

    ```sql
    CREATE TABLE table_name (
        column1 datatype DEFAULT 'some_value',
        column2 datatype,
        ...
    );
    ```
    ```sql
    ALTER TABLE table_name
    ADD CONSTRAINT DF_column1 DEFAULT 'new_default_value' FOR column1; -- Syntax varies
    ```

Constraints are vital for maintaining data quality and consistency in your database.

---
**Related Notes:**
* [[SQL]]
* [[SQL DDL]]
* [[Database Design]]
* [[Primary Key]]
* [[Foreign Key]]
* [[Referential Integrity Constraint]]
* [[Entity Integrity Constraint]]
* [[SQL NULL Values]]