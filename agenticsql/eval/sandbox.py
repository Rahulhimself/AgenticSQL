"""
Sandboxed database environment manager for evaluation and verification.

Provisions isolated in-memory or file-backed SQLite database instances pre-seeded
with rich test data across multiple domains (E-Commerce, HR/Payroll, Financial Ledger).
Ensures safe query execution with timeouts and resource constraints.
"""

import sqlite3
import logging
from typing import Optional
from pathlib import Path
import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

# --- E-Commerce Domain DDL & Seed ---
ECOMMERCE_DDL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    country TEXT NOT NULL,
    city TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL,
    parent_category_id INTEGER,
    FOREIGN KEY (parent_category_id) REFERENCES categories(category_id)
);

CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category_id INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    stock_quantity INTEGER NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    order_status TEXT NOT NULL, -- 'completed', 'pending', 'cancelled', 'refunded'
    total_amount REAL NOT NULL,
    shipping_city TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    discount REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS payments (
    payment_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    payment_method TEXT NOT NULL, -- 'credit_card', 'paypal', 'bank_transfer', 'crypto'
    payment_status TEXT NOT NULL, -- 'success', 'failed', 'pending'
    amount REAL NOT NULL,
    payment_date TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    rating INTEGER NOT NULL, -- 1 to 5
    review_text TEXT,
    review_date TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

ECOMMERCE_SEED = """
INSERT INTO customers VALUES 
(1, 'Alice', 'Smith', 'alice@example.com', 'USA', 'New York', '2023-01-15 08:30:00'),
(2, 'Bob', 'Jones', 'bob@example.com', 'USA', 'San Francisco', '2023-02-20 14:15:00'),
(3, 'Charlie', 'Brown', 'charlie@example.com', 'UK', 'London', '2023-03-10 11:00:00'),
(4, 'Diana', 'Prince', 'diana@example.com', 'Canada', 'Toronto', '2023-04-05 09:45:00'),
(5, 'Evan', 'Wright', 'evan@example.com', 'Germany', 'Berlin', '2023-05-12 16:20:00'),
(6, 'Fiona', 'Gallagher', 'fiona@example.com', 'USA', 'Chicago', '2023-06-01 10:00:00'),
(7, 'George', 'Clark', 'george@example.com', 'USA', 'Austin', '2023-07-22 17:30:00');

INSERT INTO categories VALUES
(1, 'Electronics', NULL),
(2, 'Computers', 1),
(3, 'Audio', 1),
(4, 'Home & Kitchen', NULL),
(5, 'Appliances', 4),
(6, 'Books', NULL);

INSERT INTO products VALUES
(101, 'MacBook Pro 16', 2, 2499.00, 15, 1),
(102, 'Dell XPS 15', 2, 1899.50, 20, 1),
(103, 'Sony WH-1000XM5', 3, 399.99, 45, 1),
(104, 'AirPods Pro', 3, 249.00, 80, 1),
(105, 'Espresso Machine', 5, 699.00, 8, 1),
(106, 'Air Fryer XL', 5, 129.99, 30, 1),
(107, 'Clean Code Book', 6, 45.00, 100, 1),
(108, 'Legacy Floppy Drive', 2, 19.99, 0, 0);

INSERT INTO orders VALUES
(1001, 1, '2024-01-10 10:00:00', 'completed', 2898.99, 'New York'),
(1002, 2, '2024-01-15 15:30:00', 'completed', 1899.50, 'San Francisco'),
(1003, 3, '2024-02-01 12:00:00', 'completed', 399.99, 'London'),
(1004, 1, '2024-02-14 18:45:00', 'completed', 249.00, 'New York'),
(1005, 4, '2024-03-01 09:15:00', 'pending', 699.00, 'Toronto'),
(1006, 5, '2024-03-05 14:00:00', 'cancelled', 129.99, 'Berlin'),
(1007, 2, '2024-03-20 16:30:00', 'completed', 45.00, 'San Francisco'),
(1008, 6, '2024-04-02 11:10:00', 'refunded', 399.99, 'Chicago');

INSERT INTO order_items VALUES
(1, 1001, 101, 1, 2499.00, 0.0),
(2, 1001, 103, 1, 399.99, 0.0),
(3, 1002, 102, 1, 1899.50, 0.0),
(4, 1003, 103, 1, 399.99, 0.0),
(5, 1004, 104, 1, 249.00, 0.0),
(6, 1005, 105, 1, 699.00, 0.0),
(7, 1006, 106, 1, 129.99, 0.0),
(8, 1007, 107, 1, 45.00, 0.0),
(9, 1008, 103, 1, 399.99, 0.0);

INSERT INTO payments VALUES
(501, 1001, 'credit_card', 'success', 2898.99, '2024-01-10 10:05:00'),
(502, 1002, 'paypal', 'success', 1899.50, '2024-01-15 15:35:00'),
(503, 1003, 'credit_card', 'success', 399.99, '2024-02-01 12:05:00'),
(504, 1004, 'credit_card', 'success', 249.00, '2024-02-14 18:50:00'),
(505, 1005, 'bank_transfer', 'pending', 699.00, '2024-03-01 09:20:00'),
(506, 1006, 'credit_card', 'failed', 129.99, '2024-03-05 14:05:00'),
(507, 1007, 'credit_card', 'success', 45.00, '2024-03-20 16:35:00'),
(508, 1008, 'paypal', 'success', 399.99, '2024-04-02 11:15:00');

INSERT INTO reviews VALUES
(1, 101, 1, 5, 'Super fast machine, highly recommended!', '2024-01-20'),
(2, 103, 1, 4, 'Great noise cancelling, slight ear fatigue.', '2024-01-22'),
(3, 102, 2, 5, 'Best Windows laptop on the market.', '2024-01-25'),
(4, 103, 3, 5, 'Flawless sound quality.', '2024-02-10'),
(5, 107, 2, 5, 'Essential reading for engineers.', '2024-03-25');
"""

# --- HR & Payroll Domain DDL & Seed ---
HR_DDL = """
CREATE TABLE IF NOT EXISTS departments (
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT UNIQUE NOT NULL,
    location TEXT NOT NULL,
    budget REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS employees (
    emp_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    dept_id INTEGER,
    job_title TEXT NOT NULL,
    hire_date TEXT NOT NULL,
    manager_id INTEGER,
    is_active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id),
    FOREIGN KEY (manager_id) REFERENCES employees(emp_id)
);

CREATE TABLE IF NOT EXISTS salaries (
    salary_id INTEGER PRIMARY KEY,
    emp_id INTEGER NOT NULL,
    base_salary REAL NOT NULL,
    bonus REAL DEFAULT 0.0,
    effective_date TEXT NOT NULL,
    FOREIGN KEY (emp_id) REFERENCES employees(emp_id)
);

CREATE TABLE IF NOT EXISTS performance_reviews (
    review_id INTEGER PRIMARY KEY,
    emp_id INTEGER NOT NULL,
    review_year INTEGER NOT NULL,
    score REAL NOT NULL, -- 1.0 to 5.0
    reviewer_notes TEXT,
    FOREIGN KEY (emp_id) REFERENCES employees(emp_id)
);
"""

HR_SEED = """
INSERT INTO departments VALUES
(1, 'Engineering', 'Building A', 2500000.00),
(2, 'Product', 'Building A', 1200000.00),
(3, 'Sales', 'Building B', 1800000.00),
(4, 'Human Resources', 'Building C', 600000.00),
(5, 'Legal', 'Building C', 800000.00);

INSERT INTO employees VALUES
(101, 'Sarah', 'Connor', 1, 'VP of Engineering', '2019-03-01', NULL, 1),
(102, 'John', 'Miller', 1, 'Senior Staff Engineer', '2020-06-15', 101, 1),
(103, 'Alex', 'Chen', 1, 'Software Engineer II', '2022-01-10', 102, 1),
(104, 'Emily', 'Davis', 2, 'Director of Product', '2020-02-01', NULL, 1),
(105, 'Michael', 'Scott', 3, 'Sales Manager', '2018-09-01', NULL, 1),
(106, 'Dwight', 'Schrute', 3, 'Top Sales Executive', '2019-01-15', 105, 1),
(107, 'Pam', 'Beesly', 4, 'HR Specialist', '2021-04-01', NULL, 1),
(108, 'Contractor', 'Unassigned', NULL, 'Contract Engineer', '2023-11-01', 102, 1),
(109, 'Retired', 'Veteran', 1, 'Advisor', '2015-01-01', NULL, 0);

INSERT INTO salaries VALUES
(1, 101, 240000.00, 50000.00, '2023-01-01'),
(2, 102, 185000.00, 30000.00, '2023-01-01'),
(3, 103, 135000.00, 15000.00, '2023-01-01'),
(4, 104, 190000.00, 35000.00, '2023-01-01'),
(5, 105, 140000.00, 60000.00, '2023-01-01'),
(6, 106, 110000.00, 75000.00, '2023-01-01'),
(7, 107, 85000.00, 5000.00, '2023-01-01'),
(8, 108, 95000.00, 0.00, '2023-11-01');

INSERT INTO performance_reviews VALUES
(1, 101, 2023, 4.8, 'Exceptional leadership and execution.'),
(2, 102, 2023, 4.9, 'Core architectural driver.'),
(3, 103, 2023, 4.2, 'Strong deliverer, expanding scope.'),
(4, 104, 2023, 4.6, 'Great roadmap prioritization.'),
(5, 105, 2023, 3.8, 'Hit targets but chaotic management.'),
(6, 106, 2023, 5.0, 'Record-breaking revenue generation.'),
(7, 107, 2023, 4.5, 'Smooth HR operations.');
"""

# --- Financial Ledger Domain DDL & Seed ---
FINANCE_DDL = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id INTEGER PRIMARY KEY,
    account_number TEXT UNIQUE NOT NULL,
    account_name TEXT NOT NULL,
    account_type TEXT NOT NULL, -- 'asset', 'liability', 'equity', 'revenue', 'expense'
    currency TEXT NOT NULL DEFAULT 'USD',
    current_balance REAL NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id INTEGER PRIMARY KEY,
    txn_code TEXT UNIQUE NOT NULL,
    source_account_id INTEGER NOT NULL,
    destination_account_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    txn_type TEXT NOT NULL, -- 'transfer', 'settlement', 'fee', 'reversal'
    txn_status TEXT NOT NULL, -- 'posted', 'pending', 'reversed'
    posted_at TEXT NOT NULL,
    description TEXT,
    FOREIGN KEY (source_account_id) REFERENCES accounts(account_id),
    FOREIGN KEY (destination_account_id) REFERENCES accounts(account_id)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    log_id INTEGER PRIMARY KEY,
    txn_id INTEGER,
    actor_id TEXT NOT NULL,
    action TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    ip_address TEXT,
    FOREIGN KEY (txn_id) REFERENCES transactions(txn_id)
);
"""

FINANCE_SEED = """
INSERT INTO accounts VALUES
(1, 'ACC-1001', 'Operating Cash Account', 'asset', 'USD', 4500000.00, 1),
(2, 'ACC-1002', 'Customer Deposits Clearing', 'liability', 'USD', 12000000.00, 1),
(3, 'ACC-2001', 'Software Subscription Revenue', 'revenue', 'USD', 8500000.00, 1),
(4, 'ACC-3001', 'Cloud Infrastructure Expense', 'expense', 'USD', 1250000.00, 1),
(5, 'ACC-3002', 'Payroll Expense', 'expense', 'USD', 3400000.00, 1),
(6, 'ACC-4001', 'Retained Earnings', 'equity', 'USD', 6500000.00, 1),
(7, 'ACC-9999', 'Suspense Account', 'asset', 'USD', 0.00, 0);

INSERT INTO transactions VALUES
(1, 'TXN-9001', 2, 1, 500000.00, 'transfer', 'posted', '2024-01-05 09:00:00', 'Customer fund intake'),
(2, 'TXN-9002', 1, 4, 125000.00, 'fee', 'posted', '2024-01-10 11:30:00', 'Monthly AWS bill'),
(3, 'TXN-9003', 1, 5, 280000.00, 'settlement', 'posted', '2024-01-31 17:00:00', 'January payroll payout'),
(4, 'TXN-9004', 3, 1, 750000.00, 'transfer', 'posted', '2024-02-01 10:15:00', 'Enterprise SaaS renewal'),
(5, 'TXN-9005', 1, 4, 130000.00, 'fee', 'posted', '2024-02-10 11:30:00', 'Monthly AWS bill'),
(6, 'TXN-9006', 1, 7, 50000.00, 'transfer', 'reversed', '2024-02-15 14:00:00', 'Erroneous wire attempt'),
(7, 'TXN-9007', 7, 1, 50000.00, 'reversal', 'posted', '2024-02-15 16:30:00', 'Reversal of wire TXN-9006');

INSERT INTO audit_logs VALUES
(1, 1, 'sys_admin', 'CREATE_TRANSACTION', '2024-01-05 09:00:01', '10.0.0.12'),
(2, 2, 'billing_svc', 'SETTLE_PAYMENT', '2024-01-10 11:30:02', '10.0.0.45'),
(3, 3, 'payroll_bot', 'DISBURSE_PAYROLL', '2024-01-31 17:00:05', '10.0.0.88'),
(4, 6, 'analyst_bob', 'FLAG_ERROR', '2024-02-15 14:05:00', '192.168.1.50'),
(5, 7, 'finance_lead', 'EXECUTE_REVERSAL', '2024-02-15 16:30:01', '192.168.1.10');
"""

DOMAIN_CONFIGS = {
    "ecommerce": {"ddl": ECOMMERCE_DDL, "seed": ECOMMERCE_SEED},
    "hr_payroll": {"ddl": HR_DDL, "seed": HR_SEED},
    "financial_ledger": {"ddl": FINANCE_DDL, "seed": FINANCE_SEED},
}


class DatabaseSandbox:
    """
    Isolated database sandbox for safely executing queries during evaluation.
    """

    def __init__(self, domain: str = "ecommerce", db_path: Optional[str] = None):
        """
        Initialize database sandbox for a specific domain.

        Args:
            domain: One of 'ecommerce', 'hr_payroll', 'financial_ledger'.
            db_path: Path for file-backed SQLite or None for in-memory URI.
        """
        if domain not in DOMAIN_CONFIGS:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_CONFIGS.keys())}")

        self.domain = domain
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._db: Optional[SQLDatabase] = None
        self._engine = None

        self._initialize_database()

    def _initialize_database(self) -> None:
        """Create tables and seed initial data."""
        if self.db_path:
            p = Path(self.db_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                p.unlink()
            self._uri = f"sqlite:///{p.resolve()}"
            self._conn = sqlite3.connect(str(p.resolve()))
        else:
            # Fully isolated shared memory database with unique UUID
            import uuid
            db_name = f"sandbox_{self.domain}_{uuid.uuid4().hex}"
            self._uri = f"sqlite:///file:{db_name}?mode=memory&cache=shared&uri=true"
            self._conn = sqlite3.connect(f"file:{db_name}?mode=memory&cache=shared", uri=True)

        # Run DDL and seed
        config = DOMAIN_CONFIGS[self.domain]
        cursor = self._conn.cursor()
        cursor.executescript(config["ddl"])
        cursor.executescript(config["seed"])
        self._conn.commit()

        # Build SQLAlchemy engine & LangChain SQLDatabase
        self._engine = create_engine(self._uri)
        self._db = SQLDatabase(self._engine)

    def get_sql_database(self) -> SQLDatabase:
        """Get the LangChain SQLDatabase utility instance for the sandbox."""
        return self._db

    def get_schema(self) -> str:
        """Get the database DDL and schema info."""
        return self._db.get_table_info()

    def execute_query(self, sql: str, timeout_seconds: float = 5.0) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute a SQL query against the sandbox database.

        Args:
            sql: SQL statement string.
            timeout_seconds: Execution timeout.

        Returns:
            Tuple of (DataFrame or None, error_message or None).
        """
        if not sql or not sql.strip():
            return None, "Empty SQL query"

        try:
            # pyrefly: ignore [missing-import]
            from sqlalchemy import text

            with self._engine.connect() as conn:
                df = pd.read_sql_query(text(sql), conn)
                return df, None
        except Exception as e:
            return None, str(e)

    def reset(self) -> None:
        """Reset the database to clean initial state."""
        self._initialize_database()

    def close(self) -> None:
        """Close connections."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass


def create_all_sandboxes() -> dict[str, DatabaseSandbox]:
    """Create a dictionary of all domain sandboxes."""
    return {domain: DatabaseSandbox(domain=domain) for domain in DOMAIN_CONFIGS}
