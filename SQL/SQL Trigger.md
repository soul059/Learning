---
tags:
  - sql
  - trigger
  - database-automation
  - ddl
  - dml
---

# SQL Trigger

[[SQL|A trigger]] is a special type of stored procedure that automatically executes (fires) when a specific event occurs in the database. Triggers are associated with a specific table and are invoked in response to `INSERT`, `UPDATE`, or `DELETE` operations on that table.

Triggers are part of the database schema and are defined using [[SQL DDL]] statements.

## Purpose of Triggers

Triggers are used to automate actions or enforce complex business rules that cannot be easily handled by standard [[SQL Constraints|integrity constraints]]. Common use cases include:

* **Auditing:** Recording changes made to data (who changed what, when, old/new values) into an audit log table.
* **Enforcing Complex Business Rules:** Implementing rules that involve checking data across multiple tables or performing conditional logic before/after a data modification.
* **Maintaining Derived Data:** Automatically updating summary or derived data in other tables when base data changes.
* **Cascading Operations:** Performing actions on other tables in response to a change (though [[SQL Constraints|Foreign Key]] `ON DELETE`/`ON UPDATE` actions are often preferred for simple cascades).
* **Preventing Invalid Operations:** Rolling back a transaction if a proposed data modification violates a complex rule.

## Trigger Syntax (Conceptual)

The exact syntax for creating triggers varies significantly between [[DBMS|Database Management Systems]] (e.g., SQL Server, Oracle, PostgreSQL, MySQL). However, the core components are similar:

```sql
CREATE TRIGGER trigger_name
ON table_name
AFTER | BEFORE | INSTEAD OF -- When the trigger fires (timing)
INSERT | UPDATE | DELETE     -- What event(s) cause the trigger to fire
[FOR EACH ROW | FOR EACH STATEMENT] -- Granularity (row-level or statement-level)
AS
BEGIN
    -- SQL statements or procedural code to execute
    -- Can access 'old' and 'new' data (syntax varies)
END;
```

* `trigger_name`: A unique name for the trigger.
* `table_name`: The table the trigger is associated with.
* **Timing:**
    * `AFTER`: Fires after the `INSERT`, `UPDATE`, or `DELETE` operation is completed.
    * `BEFORE`: Fires before the `INSERT`, `UPDATE`, or `DELETE` operation is performed.
    * `INSTEAD OF`: (Common in SQL Server) Fires instead of the actual `INSERT`, `UPDATE`, or `DELETE` operation, often used for views.
* **Event(s):** Specifies which DML operation(s) will fire the trigger.
* **Granularity:**
    * `FOR EACH ROW`: The trigger body executes once for each row affected by the DML statement.
    * `FOR EACH STATEMENT`: The trigger body executes once for the entire DML statement, regardless of how many rows are affected.
* **Trigger Body (`AS BEGIN...END`):** Contains the [[SQL|SQL]] or procedural code that runs when the trigger fires. Within the trigger body, you can usually access the data being inserted, updated, or deleted (e.g., using logical tables like `INSERTED` and `DELETED` in SQL Server, or `:OLD` and `:NEW` row variables in Oracle/PostgreSQL).

## Example (Conceptual - SQL Server Syntax)

```sql
-- Create an audit table
CREATE TABLE OrderAudit (
    AuditID INT IDENTITY PRIMARY KEY,
    OrderID INT,
    ChangeType VARCHAR(10), -- 'INSERT', 'UPDATE', 'DELETE'
    OldAmount DECIMAL(10,2),
    NewAmount DECIMAL(10,2),
    ChangeDate DATETIME DEFAULT GETDATE(),
    ChangedBy VARCHAR(100) DEFAULT SUSER_SNAME() -- Get current user
);
```sql
-- Create a trigger to log updates to the Orders table Amount
CREATE TRIGGER trg_OrderAmountUpdate
ON Orders
AFTER UPDATE
AS
BEGIN
    -- Check if the Amount column was actually updated
    IF UPDATE(Amount)
    BEGIN
        -- Insert a row into the audit table for each updated order
        INSERT INTO OrderAudit (OrderID, ChangeType, OldAmount, NewAmount)
        SELECT
            i.OrderID,
            'UPDATE',
            d.Amount, -- Old value from the DELETED pseudo-table
            i.Amount  -- New value from the INSERTED pseudo-table
        FROM
            INSERTED i -- Contains the new rows after update
        INNER JOIN
            DELETED d ON i.OrderID = d.OrderID; -- Contains the old rows before update
    END
END;
```

## Disadvantages of Triggers

While powerful, triggers should be used judiciously:

* **Hidden Logic:** Trigger logic is not immediately visible when examining simple [[SQL DML]] statements. This can make debugging and understanding data flow difficult.
* **Performance Impact:** Triggers execute synchronously with the DML operation. Complex trigger logic can significantly slow down inserts, updates, and deletes.
* **Cascading Effects:** A trigger on one table might fire a trigger on another table, leading to complex and potentially hard-to-follow cascading operations.
* **DBMS-Specific Syntax:** Trigger syntax and behavior vary widely, making code less portable.

Triggers are best used for cross-cutting concerns like auditing or enforcing complex, non-negotiable business rules that must fire automatically. For application-specific logic, it's often better to implement rules in the application layer or within explicit stored procedures called by the application.

---
**Related Notes:**
* [[SQL]]
* [[SQL DDL]]
* [[SQL DML]]
* [[SQL Constraints]]
* [[Stored Procedures]] (Create if needed)
* [[Database Automation]] (Create if needed)
* [[Database Security]] (Related to auditing)