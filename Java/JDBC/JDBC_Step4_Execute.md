# JDBC Step 4: Executing SQL Statements - Query Execution Methods

Once you have created a `Statement` or `PreparedStatement` object, you need to execute it to interact with the database. JDBC provides different execution methods depending on the type of SQL operation you're performing and the expected results.

---

## SQL Statement Categories

Before diving into execution methods, it's important to understand the different categories of SQL statements:

### 1. **Data Query Language (DQL)**
- **Purpose**: Retrieve data from the database
- **Primary Command**: `SELECT`
- **Returns**: Result set with data
- **JDBC Method**: `executeQuery()`

### 2. **Data Manipulation Language (DML)**
- **Purpose**: Modify data in the database
- **Commands**: `INSERT`, `UPDATE`, `DELETE`
- **Returns**: Number of affected rows
- **JDBC Method**: `executeUpdate()`

### 3. **Data Definition Language (DDL)**
- **Purpose**: Define or modify database structure
- **Commands**: `CREATE`, `ALTER`, `DROP`, `TRUNCATE`
- **Returns**: Usually 0 (success) or throws exception
- **JDBC Method**: `executeUpdate()` or `execute()`

### 4. **Data Control Language (DCL)**
- **Purpose**: Control access permissions
- **Commands**: `GRANT`, `REVOKE`
- **Returns**: Usually 0 (success) or throws exception
- **JDBC Method**: `executeUpdate()`

---

## JDBC Execution Methods

JDBC provides three primary execution methods, each designed for specific SQL statement types:

### 1. `executeQuery()` - For SELECT Statements

This method is exclusively used for SQL statements that return data (typically `SELECT` statements).

**Characteristics:**
- **Return Type**: `ResultSet` object containing the query results
- **Use Case**: When you need to retrieve data from the database
- **Throws**: `SQLException` if the SQL is not a query or if an error occurs

#### Basic Usage Example

```java
public class QueryExecutionExample {
    
    public void demonstrateSimpleQuery(Connection conn) throws SQLException {
        String sql = "SELECT id, name, email, salary FROM employees WHERE department_id = ?";
        
        try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, 10); // Set department_id parameter
            
            try (ResultSet rs = pstmt.executeQuery()) {
                System.out.printf("%-5s %-20s %-30s %-10s%n", "ID", "Name", "Email", "Salary");
                System.out.println("-".repeat(70));
                
                while (rs.next()) {
                    int id = rs.getInt("id");
                    String name = rs.getString("name");
                    String email = rs.getString("email");
                    BigDecimal salary = rs.getBigDecimal("salary");
                    
                    System.out.printf("%-5d %-20s %-30s $%-9.2f%n", 
                        id, name, email, salary);
                }
            }
        }
    }
}
```

#### Complex Query with Joins

```java
public List<EmployeeDetails> getEmployeeDetailsWithDepartment(Connection conn, int departmentId) 
        throws SQLException {
    
    String sql = """
        SELECT 
            e.id, e.name, e.email, e.salary, e.hire_date,
            d.name as department_name, d.location as department_location,
            m.name as manager_name
        FROM employees e
        JOIN departments d ON e.department_id = d.id
        LEFT JOIN employees m ON e.manager_id = m.id
        WHERE e.department_id = ?
        ORDER BY e.name
        """;
    
    List<EmployeeDetails> employees = new ArrayList<>();
    
    try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
        pstmt.setInt(1, departmentId);
        
        try (ResultSet rs = pstmt.executeQuery()) {
            while (rs.next()) {
                EmployeeDetails employee = new EmployeeDetails(
                    rs.getInt("id"),
                    rs.getString("name"),
                    rs.getString("email"),
                    rs.getBigDecimal("salary"),
                    rs.getDate("hire_date").toLocalDate(),
                    rs.getString("department_name"),
                    rs.getString("department_location"),
                    rs.getString("manager_name")
                );
                employees.add(employee);
            }
        }
    }
    
    return employees;
}
```

---

### 2. `executeUpdate()` - For DML and DDL Statements

This method is used for SQL statements that modify data or database structure but don't return a result set.

**Characteristics:**
- **Return Type**: `int` representing the number of affected rows
- **Use Cases**: `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `ALTER`, `DROP`
- **Special Cases**: DDL statements typically return 0

#### INSERT Operations

```java
public class InsertOperationExample {
    
    // Insert single record with generated keys
    public long insertEmployee(Connection conn, Employee employee) throws SQLException {
        String sql = """
            INSERT INTO employees (name, email, salary, department_id, hire_date) 
            VALUES (?, ?, ?, ?, CURRENT_DATE)
            """;
        
        try (PreparedStatement pstmt = conn.prepareStatement(sql, Statement.RETURN_GENERATED_KEYS)) {
            
            pstmt.setString(1, employee.getName());
            pstmt.setString(2, employee.getEmail());
            pstmt.setBigDecimal(3, employee.getSalary());
            pstmt.setInt(4, employee.getDepartmentId());
            
            int affectedRows = pstmt.executeUpdate();
            
            if (affectedRows == 0) {
                throw new SQLException("Creating employee failed, no rows affected.");
            }
            
            try (ResultSet generatedKeys = pstmt.getGeneratedKeys()) {
                if (generatedKeys.next()) {
                    long employeeId = generatedKeys.getLong(1);
                    System.out.println("Employee created with ID: " + employeeId);
                    return employeeId;
                } else {
                    throw new SQLException("Creating employee failed, no ID obtained.");
                }
            }
        }
    }
    
    // Batch insert for multiple records
    public int[] insertEmployeesBatch(Connection conn, List<Employee> employees) throws SQLException {
        String sql = """
            INSERT INTO employees (name, email, salary, department_id, hire_date) 
            VALUES (?, ?, ?, ?, ?)
            """;
        
        try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            for (Employee employee : employees) {
                pstmt.setString(1, employee.getName());
                pstmt.setString(2, employee.getEmail());
                pstmt.setBigDecimal(3, employee.getSalary());
                pstmt.setInt(4, employee.getDepartmentId());
                pstmt.setDate(5, Date.valueOf(employee.getHireDate()));
                
                pstmt.addBatch();
            }
            
            int[] results = pstmt.executeBatch();
            System.out.println("Batch insert completed. Records inserted: " + results.length);
            return results;
        }
    }
}
```

#### UPDATE Operations

```java
public class UpdateOperationExample {
    
    // Update single record
    public boolean updateEmployeeSalary(Connection conn, int employeeId, BigDecimal newSalary) 
            throws SQLException {
        
        String sql = """
            UPDATE employees 
            SET salary = ?, last_updated = CURRENT_TIMESTAMP 
            WHERE id = ? AND active = true
            """;
        
        try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setBigDecimal(1, newSalary);
            pstmt.setInt(2, employeeId);
            
            int rowsAffected = pstmt.executeUpdate();
            
            if (rowsAffected > 0) {
                System.out.println("Employee salary updated successfully. Rows affected: " + rowsAffected);
                return true;
            } else {
                System.out.println("No active employee found with ID: " + employeeId);
                return false;
            }
        }
    }
    
    // Bulk update with conditions
    public int giveRaiseByDepartment(Connection conn, int departmentId, double raisePercentage) 
            throws SQLException {
        
        String sql = """
            UPDATE employees 
            SET salary = salary * (1 + ? / 100),
                last_updated = CURRENT_TIMESTAMP
            WHERE department_id = ? 
            AND active = true 
            AND hire_date < CURRENT_DATE - INTERVAL '6 months'
            """;
        
        try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setDouble(1, raisePercentage);
            pstmt.setInt(2, departmentId);
            
            int rowsAffected = pstmt.executeUpdate();
            System.out.printf("Gave %.1f%% raise to %d employees in department %d%n", 
                raisePercentage, rowsAffected, departmentId);
            
            return rowsAffected;
        }
    }
}
```

#### DELETE Operations

```java
public class DeleteOperationExample {
    
    // Soft delete (recommended)
    public boolean deactivateEmployee(Connection conn, int employeeId) throws SQLException {
        String sql = """
            UPDATE employees 
            SET active = false, deactivated_date = CURRENT_TIMESTAMP 
            WHERE id = ? AND active = true
            """;
        
        try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, employeeId);
            
            int rowsAffected = pstmt.executeUpdate();
            return rowsAffected > 0;
        }
    }
    
    // Hard delete with cascade check
    public boolean deleteEmployeeWithValidation(Connection conn, int employeeId) throws SQLException {
        // First check if employee has dependencies
        String checkSql = """
            SELECT COUNT(*) FROM projects_employees WHERE employee_id = ?
            UNION ALL
            SELECT COUNT(*) FROM employee_evaluations WHERE employee_id = ?
            """;
        
        try (PreparedStatement checkStmt = conn.prepareStatement(checkSql)) {
            checkStmt.setInt(1, employeeId);
            checkStmt.setInt(2, employeeId);
            
            try (ResultSet rs = checkStmt.executeQuery()) {
                int totalDependencies = 0;
                while (rs.next()) {
                    totalDependencies += rs.getInt(1);
                }
                
                if (totalDependencies > 0) {
                    throw new SQLException(
                        "Cannot delete employee: " + totalDependencies + " dependent records exist");
                }
            }
        }
        
        // Safe to delete
        String deleteSql = "DELETE FROM employees WHERE id = ?";
        try (PreparedStatement deleteStmt = conn.prepareStatement(deleteSql)) {
            deleteStmt.setInt(1, employeeId);
            
            int rowsAffected = deleteStmt.executeUpdate();
            return rowsAffected > 0;
        }
    }
}
```

#### DDL Operations

```java
public class DDLOperationExample {
    
    public void createEmployeeTable(Connection conn) throws SQLException {
        String sql = """
            CREATE TABLE IF NOT EXISTS employees (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                salary DECIMAL(12,2) CHECK (salary > 0),
                department_id INTEGER REFERENCES departments(id),
                hire_date DATE NOT NULL DEFAULT CURRENT_DATE,
                active BOOLEAN DEFAULT true,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """;
        
        try (Statement stmt = conn.createStatement()) {
            int result = stmt.executeUpdate(sql);
            System.out.println("Table 'employees' created. Result: " + result);
        }
    }
    
    public void createIndexes(Connection conn) throws SQLException {
        String[] indexes = {
            "CREATE INDEX IF NOT EXISTS idx_employees_department ON employees(department_id)",
            "CREATE INDEX IF NOT EXISTS idx_employees_email ON employees(email)",
            "CREATE INDEX IF NOT EXISTS idx_employees_active ON employees(active) WHERE active = true"
        };
        
        try (Statement stmt = conn.createStatement()) {
            for (String indexSql : indexes) {
                int result = stmt.executeUpdate(indexSql);
                System.out.println("Index created. Result: " + result);
            }
        }
    }
}
```

---

### 3. `execute()` - For Dynamic or Unknown Statement Types

This is the most general execution method that can handle any type of SQL statement.

**Characteristics:**
- **Return Type**: `boolean` indicating the type of first result
- **Returns `true`**: If first result is a `ResultSet` (query)
- **Returns `false`**: If first result is an update count or no result
- **Use Cases**: Dynamic SQL, stored procedures, multiple result sets

#### Basic Usage

```java
public class ExecuteMethodExample {
    
    public void executeUnknownSQL(Connection conn, String sql) throws SQLException {
        try (Statement stmt = conn.createStatement()) {
            boolean hasResultSet = stmt.execute(sql);
            
            if (hasResultSet) {
                // It's a query - process ResultSet
                try (ResultSet rs = stmt.getResultSet()) {
                    ResultSetMetaData metaData = rs.getMetaData();
                    int columnCount = metaData.getColumnCount();
                    
                    // Print column headers
                    for (int i = 1; i <= columnCount; i++) {
                        System.out.printf("%-15s", metaData.getColumnName(i));
                    }
                    System.out.println();
                    
                    // Print data
                    while (rs.next()) {
                        for (int i = 1; i <= columnCount; i++) {
                            System.out.printf("%-15s", rs.getString(i));
                        }
                        System.out.println();
                    }
                }
            } else {
                // It's an update - get update count
                int updateCount = stmt.getUpdateCount();
                System.out.println("Update count: " + updateCount);
            }
        }
    }
    
    // Handle multiple result sets (stored procedures)
    public void executeStoredProcedureWithMultipleResults(Connection conn, int employeeId) 
            throws SQLException {
        
        String sql = "{call get_employee_full_report(?)}"; // Returns multiple result sets
        
        try (CallableStatement cstmt = conn.prepareCall(sql)) {
            cstmt.setInt(1, employeeId);
            
            boolean hasResults = cstmt.execute();
            int resultSetCount = 0;
            
            do {
                if (hasResults) {
                    // Process current ResultSet
                    try (ResultSet rs = cstmt.getResultSet()) {
                        resultSetCount++;
                        System.out.println("Processing ResultSet #" + resultSetCount);
                        
                        ResultSetMetaData metaData = rs.getMetaData();
                        int columnCount = metaData.getColumnCount();
                        
                        while (rs.next()) {
                            for (int i = 1; i <= columnCount; i++) {
                                System.out.print(metaData.getColumnName(i) + ": " + 
                                    rs.getString(i) + " ");
                            }
                            System.out.println();
                        }
                    }
                } else {
                    // Process update count
                    int updateCount = cstmt.getUpdateCount();
                    if (updateCount != -1) {
                        System.out.println("Update count: " + updateCount);
                    }
                }
                
                // Move to next result
                hasResults = cstmt.getMoreResults();
                
            } while (hasResults || cstmt.getUpdateCount() != -1);
        }
    }
}
```

---

## Advanced Execution Patterns

### 1. Transaction-Safe Execution

```java
public class TransactionSafeExecution {
    
    public void transferMoney(Connection conn, int fromAccount, int toAccount, BigDecimal amount) 
            throws SQLException {
        
        conn.setAutoCommit(false); // Start transaction
        
        try {
            // Debit from source account
            String debitSql = """
                UPDATE accounts 
                SET balance = balance - ? 
                WHERE id = ? AND balance >= ?
                """;
            
            try (PreparedStatement debitStmt = conn.prepareStatement(debitSql)) {
                debitStmt.setBigDecimal(1, amount);
                debitStmt.setInt(2, fromAccount);
                debitStmt.setBigDecimal(3, amount);
                
                int debitRows = debitStmt.executeUpdate();
                if (debitRows == 0) {
                    throw new SQLException("Insufficient funds or account not found");
                }
            }
            
            // Credit to destination account
            String creditSql = "UPDATE accounts SET balance = balance + ? WHERE id = ?";
            
            try (PreparedStatement creditStmt = conn.prepareStatement(creditSql)) {
                creditStmt.setBigDecimal(1, amount);
                creditStmt.setInt(2, toAccount);
                
                int creditRows = creditStmt.executeUpdate();
                if (creditRows == 0) {
                    throw new SQLException("Destination account not found");
                }
            }
            
            // Log transaction
            String logSql = """
                INSERT INTO transaction_log (from_account, to_account, amount, transaction_date) 
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """;
            
            try (PreparedStatement logStmt = conn.prepareStatement(logSql)) {
                logStmt.setInt(1, fromAccount);
                logStmt.setInt(2, toAccount);
                logStmt.setBigDecimal(3, amount);
                logStmt.executeUpdate();
            }
            
            conn.commit(); // Commit transaction
            System.out.println("Money transfer completed successfully");
            
        } catch (SQLException e) {
            conn.rollback(); // Rollback on error
            System.err.println("Money transfer failed: " + e.getMessage());
            throw e;
        } finally {
            conn.setAutoCommit(true); // Restore auto-commit
        }
    }
}
```

### 2. Bulk Operations with Progress Tracking

```java
public class BulkOperationWithProgress {
    
    public void importLargeDataset(Connection conn, List<DataRecord> records, 
                                   ProgressCallback callback) throws SQLException {
        
        String sql = "INSERT INTO large_table (col1, col2, col3, col4) VALUES (?, ?, ?, ?)";
        
        try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            conn.setAutoCommit(false);
            
            int batchSize = 1000;
            int processedCount = 0;
            int totalRecords = records.size();
            
            for (DataRecord record : records) {
                pstmt.setString(1, record.getCol1());
                pstmt.setString(2, record.getCol2());
                pstmt.setInt(3, record.getCol3());
                pstmt.setBigDecimal(4, record.getCol4());
                
                pstmt.addBatch();
                processedCount++;
                
                // Execute batch and report progress
                if (processedCount % batchSize == 0) {
                    int[] results = pstmt.executeBatch();
                    pstmt.clearBatch();
                    
                    double progressPercent = (double) processedCount / totalRecords * 100;
                    callback.onProgress(processedCount, totalRecords, progressPercent);
                    
                    // Commit periodically to avoid long transactions
                    conn.commit();
                }
            }
            
            // Execute remaining records
            if (processedCount % batchSize != 0) {
                pstmt.executeBatch();
                conn.commit();
            }
            
            callback.onComplete(processedCount);
            
        } catch (SQLException e) {
            conn.rollback();
            throw new SQLException("Bulk import failed at record " + processedCount, e);
        } finally {
            conn.setAutoCommit(true);
        }
    }
    
    @FunctionalInterface
    public interface ProgressCallback {
        void onProgress(int processed, int total, double percent);
        
        default void onComplete(int totalProcessed) {
            System.out.println("Import completed. Total records: " + totalProcessed);
        }
    }
}
```

---

## Error Handling and Best Practices

### 1. Comprehensive Error Handling

```java
public class RobustExecutionExample {
    
    public void executeWithComprehensiveErrorHandling(Connection conn, String sql, Object... params) {
        try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            // Set parameters
            for (int i = 0; i < params.length; i++) {
                pstmt.setObject(i + 1, params[i]);
            }
            
            // Determine execution method based on SQL type
            if (sql.trim().toUpperCase().startsWith("SELECT")) {
                try (ResultSet rs = pstmt.executeQuery()) {
                    // Process results...
                }
            } else {
                int rowsAffected = pstmt.executeUpdate();
                System.out.println("Rows affected: " + rowsAffected);
            }
            
        } catch (SQLTimeoutException e) {
            System.err.println("Query timeout: " + e.getMessage());
            // Handle timeout specifically
        } catch (SQLException e) {
            handleSQLException(e, sql);
        }
    }
    
    private void handleSQLException(SQLException e, String sql) {
        System.err.println("SQL Error occurred while executing: " + sql);
        System.err.println("Error Code: " + e.getErrorCode());
        System.err.println("SQL State: " + e.getSQLState());
        System.err.println("Message: " + e.getMessage());
        
        // Handle specific error conditions
        switch (e.getErrorCode()) {
            case 23505: // Unique constraint violation (PostgreSQL)
                System.err.println("Duplicate record detected");
                break;
            case 23503: // Foreign key constraint violation
                System.err.println("Referenced record does not exist");
                break;
            case 23514: // Check constraint violation
                System.err.println("Data validation failed");
                break;
            default:
                System.err.println("Unexpected database error");
        }
    }
}
```

### 2. Performance Optimization Tips

```java
public class PerformanceOptimizedExecution {
    
    public void optimizedLargeResultSetProcessing(Connection conn) throws SQLException {
        String sql = "SELECT * FROM large_table WHERE status = ?";
        
        try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
            // Set fetch size for memory efficiency
            pstmt.setFetchSize(1000);
            
            // Set query timeout
            pstmt.setQueryTimeout(300); // 5 minutes
            
            pstmt.setString(1, "ACTIVE");
            
            try (ResultSet rs = pstmt.executeQuery()) {
                int rowCount = 0;
                while (rs.next()) {
                    // Process each row
                    processRow(rs);
                    
                    rowCount++;
                    if (rowCount % 10000 == 0) {
                        System.out.println("Processed " + rowCount + " rows");
                    }
                }
            }
        }
    }
    
    private void processRow(ResultSet rs) throws SQLException {
        // Efficient row processing
        // Avoid creating unnecessary objects
        // Use appropriate data type methods
    }
}
```

---

## Summary of Best Practices

1. **Choose the Right Method**: Use `executeQuery()` for SELECT, `executeUpdate()` for DML/DDL
2. **Always Use try-with-resources**: Ensures proper resource cleanup
3. **Handle Exceptions Appropriately**: Different error codes require different handling
4. **Use Transactions**: For multi-statement operations that must be atomic
5. **Set Timeouts**: Prevent long-running queries from hanging
6. **Use Batch Operations**: For better performance with multiple similar operations
7. **Set Fetch Size**: For memory-efficient processing of large result sets
8. **Validate Results**: Check return values and affected row counts
9. **Log Operations**: For debugging and audit trails
10. **Use PreparedStatement**: Always prefer over Statement for security and performance

## 2. `executeUpdate()`

This method is used for SQL statements that modify data in the database. This includes `INSERT`, `UPDATE`, and `DELETE` statements. It can also be used for Data Definition Language (DDL) statements like `CREATE TABLE`, `DROP TABLE`, and `ALTER TABLE`.

-   **Return Value**: It returns an `int`.
    -   For `INSERT`, `UPDATE`, or `DELETE`, the `int` represents the **number of rows affected** by the operation.
    -   For DDL statements, the return value is typically `0`.
-   **Usage**: Use this method when you are changing the state or structure of the database.

### Example with `PreparedStatement`

```java
String sql = "UPDATE products SET price = ? WHERE id = ?";

try (PreparedStatement pstmt = connection.prepareStatement(sql)) {
    
    pstmt.setDouble(1, 19.99); // Set the new price
    pstmt.setInt(2, 101);      // Set the product ID
    
    // Execute the UPDATE statement
    int rowsAffected = pstmt.executeUpdate();
    
    System.out.println(rowsAffected + " row(s) were updated successfully.");
    
    if (rowsAffected > 0) {
        // The update was successful
    } else {
        // The update did not affect any rows (e.g., product with ID 101 not found)
    }
    
} catch (SQLException e) {
    e.printStackTrace();
}
```

---

## 3. `execute()`

This method is more general and can handle any type of SQL statement. It's useful when you are working with a dynamic SQL statement and don't know in advance whether it will be a `SELECT` or an update.

-   **Return Value**: It returns a `boolean`.
    -   `true`: The result is a `ResultSet` (i.e., it was a `SELECT` statement).
    -   `false`: The result is an update count or there is no result (i.e., it was an `INSERT`, `UPDATE`, `DELETE`, or DDL statement).

-   **Usage**: Typically used in frameworks or tools that need to execute arbitrary SQL, but less common in day-to-day application code.

### How to Use `execute()`

If `execute()` returns `true`, you can get the `ResultSet` by calling `getResultSet()`. If it returns `false`, you can get the number of affected rows by calling `getUpdateCount()`.

### Example

```java
String dynamicSql = getDynamicSql(); // Method that returns some SQL string

try (Statement stmt = connection.createStatement()) {
    
    boolean isResultSet = stmt.execute(dynamicSql);
    
    if (isResultSet) {
        // It was a SELECT statement
        try (ResultSet rs = stmt.getResultSet()) {
            System.out.println("Query returned a result set:");
            while (rs.next()) {
                // ... process results
            }
        }
    } else {
        // It was an update or DDL statement
        int updateCount = stmt.getUpdateCount();
        if (updateCount != -1) {
            System.out.println("Query was an update. " + updateCount + " row(s) affected.");
        } else {
            System.out.println("Query did not return a result set or an update count.");
        }
    }
    
} catch (SQLException e) {
    e.printStackTrace();
}
```

## Summary

| Method            | Used For                               | Returns                               |
|-------------------|----------------------------------------|---------------------------------------|
| `executeQuery()`  | `SELECT` statements                    | `ResultSet`                           |
| `executeUpdate()` | `INSERT`, `UPDATE`, `DELETE`, DDL      | `int` (number of rows affected)       |
| `execute()`       | Any type of SQL statement              | `boolean` (true if `ResultSet`)       |

For most application development, you will primarily use `executeQuery()` for reading data and `executeUpdate()` for modifying data.
