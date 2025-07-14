# Linux Permission Management

In Linux, permissions are a fundamental security mechanism that controls who can access and perform actions on files and directories. Permissions are assigned to a file's owner, its associated group, and all other users on the system. Understanding and managing these permissions is crucial for system security and proper collaboration.

## Basic Permissions

Every file and directory has three basic types of permissions:

* **r (Read):**
    * For files: Allows viewing the contents of the file.
    * For directories: Allows listing the contents of the directory.
* **w (Write):**
    * For files: Allows modifying the content of the file.
    * For directories: Allows creating, deleting, or renaming files and subdirectories within the directory.
* **x (Execute):**
    * For files: Allows running the file as a program or script.
    * For directories: Allows **traversing** or entering the directory. This is necessary to access anything inside, even if you have read permission to list contents.

These permissions are applied to three categories of users:

1.  **Owner (`u`)**: The user who owns the file or directory.
2.  **Group (`g`)**: Users who are members of the file's or directory's associated group.
3.  **Others (`o`)**: All other users on the system.
4.  **All (`a`)**: Refers to owner, group, and others.

## Viewing Permissions (`ls -l`)

The `ls -l` command displays file and directory information in a long format, including the permission string.

* **Example Output Structure:**
![[Permission Management.png]]
    ```
    - rwx rw- r--   1 owner group   size date time filename
    ```
* **Permission String Breakdown (`-rwxrw-r--`):**
    1.  **First Character (`-`)**: Indicates the file type (`-` for a regular file, `d` for a directory, `l` for a symbolic link, etc.).
    2.  **Next three characters (`rwx`)**: Permissions for the **owner**.
    3.  **Next three characters (`rw-`)**: Permissions for the **group**.
    4.  **Last three characters (`r--`)**: Permissions for **others**.
    * A hyphen (`-`) means the permission is *not* granted for that category.

* **Directory Execute (`x`) Permission:** Crucially, the `x` permission on a directory is required to enter or traverse it. Without it, you will get "Permission Denied" even if you can list its contents (`r`).

## Changing Permissions (`chmod`)

The `chmod` command is used to modify the permissions of files and directories.

* **Methods:**
    * **Symbolic Method:** Use user references (`u`, `g`, `o`, `a`), operators (`+` to add, `-` to remove, `=` to set explicitly), and permission types (`r`, `w`, `x`).
        * **Example (Add read permission for all users):**
            ```bash
            chmod a+r shell
            ```
    * **Octal Method:** Use a three-digit octal number where each digit represents the combined permissions for owner, group, and others.
        * **Octal Values:**
            * Read (`r`): 4
            * Write (`w`): 2
            * Execute (`x`): 1
        * Sum the values for the desired permissions:
            * `rwx` = 4 + 2 + 1 = 7
            * `rw-` = 4 + 2 + 0 = 6
            * `r-x` = 4 + 0 + 1 = 5
            * `r--` = 4 + 0 + 0 = 4
            * `-wx` = 0 + 2 + 1 = 3
            * `-w-` = 0 + 2 + 0 = 2
            * `--x` = 0 + 0 + 1 = 1
            * `---` = 0 + 0 + 0 = 0
        * **Example (Set owner=rwx, group=r-x, others=r--):**
            ```bash
            chmod 754 shell
            ```

## Changing Ownership (`chown`)

The `chown` command changes the owner and/or group of a file or directory.

* **Syntax:**
    ```bash
    chown %3Cuser%3E:<group> <file/directory>
    ```
* **Example (Change owner and group to root):**
    ```bash
    chown root:root shell
    ```
* You can change only the owner (`chown <user> <file>`) or only the group (`chown :<group> <file>`).

## Special Permissions (SUID, SGID, Sticky Bit)

Linux has special permission bits that grant specific elevated privileges.

* **Set User ID (SUID) & Set Group ID (SGID):**
    * **Purpose:** When set on an executable file, allows the program to run with the permissions of the file's *owner* (SUID) or *group* (SGID), rather than the executing user.
    * **Indication in `ls -l`:** An `s` appears in the owner's `x` position (SUID) or group's `x` position (SGID). If the original execute permission was not set, a capital `S` is shown.
    * **Security Risk:** If applied to programs that can spawn a shell or perform sensitive actions, it can allow privilege escalation (e.g., a non-root user running a SUID-enabled program that then runs a shell as root). Resources like [GTFOBins](https://gtfobins.github.io/) document common binaries with exploitable SUID/SGID permissions.

* **Sticky Bit:**
    * **Purpose:** When set on a *directory*, it prevents users from deleting or renaming files within that directory unless they are the file's owner, the directory's owner, or the root user.
    * **Use Case:** Commonly used on shared directories (like `/tmp`) to allow multiple users to write files but prevent them from deleting others' files.
    * **Indication in `ls -l`:** A `t` appears in the others' `x` position. If the original execute permission for others was not set, a capital `T` is shown.
    * **Difference between `t` and `T`:**
        * `t`: Sticky bit is set, AND others have execute permission on the directory.
        * `T` : Sticky bit is set, BUT others do *not* have execute permission on the directory.

Understanding these permission concepts and commands is crucial for securing Linux systems and effectively managing file access.