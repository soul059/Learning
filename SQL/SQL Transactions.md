---
tags:
  - sql
  - transactions
  - tcl
  - acid
  - database-integrity
---

# SQL Transactions

[[SQL|A transaction]] is a single logical unit of work in a database. It's a sequence of one or more SQL statements that are treated as a single, atomic operation. This is crucial for maintaining data consistency and reliability, especially in multi-user environments or in the event of system failures.

The concept of transactions is deeply tied to the [[Transaction Management]] component of a [[DBMS]] and is designed to adhere to the [[ACID Properties]] (Atomicity, Consistency, Isolation, Durability).

## ACID Properties

While the detailed explanation of ACID properties might be in a more fundamental [[Database Concepts]] note, in the context of SQL Transactions:

* **Atomicity:** The transaction is treated as an indivisible unit. Either all of its operations are completed successfully and committed to the database, or if any part fails, the entire transaction is aborted, and the database is rolled back to its state before the transaction began. ("All or nothing").
* **Consistency:** A transaction brings the database from one valid state to another valid state. It ensures that the database obeys all defined integrity constraints ([[SQL Constraints]]). The transaction itself doesn't violate these rules.
* **Isolation:** Concurrent transactions execute in isolation from each other. Each transaction appears to run as if it were the only operation on the database, preventing interference and anomalies (like dirty reads, phantom reads, non-repeatable reads) that can occur when transactions access shared data concurrently.
* **Durability:** Once a transaction has been successfully committed, its changes are permanent and will survive even system failures (like power outages, crashes). The DBMS ensures that committed data is written to persistent storage.

## SQL TCL (Transaction Control Language)

SQL provides commands to manage transactions:

* `START TRANSACTION` or `BEGIN TRANSACTION` or `BEGIN`: Starts a new transaction. All subsequent SQL statements until a `COMMIT` or `ROLLBACK` will be part of this transaction.
* `COMMIT`: Saves all changes made during the current transaction permanently to the database. The transaction ends.
* `ROLLBACK`: Undoes all changes made during the current transaction and restores the database to the state it was in before the transaction started. The transaction ends.
* `SAVEPOINT savepoint_name`: Sets a marker within a transaction. You can then `ROLLBACK` to this savepoint instead of rolling back the entire transaction.

**Example Scenario (Transferring Money):**

Let's say we need to transfer $100 from Account A to Account B. This involves two steps:
1.  Debit Account A by $100.
2.  Credit Account B by $100.

If only step 1 completes and the system crashes before step 2, the database is in an inconsistent state. A transaction ensures both steps happen or neither happens.

```sql
START TRANSACTION; -- Or BEGIN;

-- Step 1: Debit Account A
UPDATE Accounts
SET Balance = Balance - 100
WHERE AccountNumber = 'AccountA';

-- Check if debit was successful (optional, but good practice)
-- If debit fails (e.g., insufficient funds, though CHECK constraint is better)
-- ROLLBACK;
-- SELECT 'Transfer failed due to insufficient funds';

-- Step 2: Credit Account B
UPDATE Accounts
SET Balance = Balance + 100
WHERE AccountNumber = 'AccountB';

-- If both steps completed successfully
COMMIT; -- Make changes permanent

-- If something went wrong after step 1 but before step 2
-- ROLLBACK; -- Undo the debit to Account A
-- SELECT 'Transfer failed due to system error';
```

* **Autocommit:** Most database systems have an autocommit mode, where each individual SQL statement (`INSERT`, `UPDATE`, `DELETE`) is automatically committed as a separate transaction if it executes successfully. You explicitly use `START TRANSACTION` to group multiple statements into a single transaction.

Transactions are a cornerstone of database reliability, ensuring that your data remains consistent and correct even in the face of errors or concurrent access.

---
**Related Notes:**
* [[SQL]]
* [[SQL DML]]
* [[Transaction Management]]
* [[ACID Properties]] (create this note if you want a deeper dive into ACID itself)
* [[Database Concepts]] (link to your main note if it covers ACID)
```

```markdown
***
**End of File 12: `SQL Transactions.md`**
***
```

---

I have added separators like `*** Start of File X ***` and `*** End of File X ***` to clearly delineate the content for each file in the output. Please copy the content *between* these markers (including the `---` YAML delimiters at the top and bottom of each file) into the respective Markdown files in your Obsidian vault.

I have reviewed the formatting, especially code blocks and tables, to ensure better compatibility with standard Markdown and Obsidian's rendering engine. Hopefully, this resolves the formatting issues you encountered previously.