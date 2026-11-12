-- NyumbaAgent analysis database (docs/03_DATA_MODEL.md). Money is integer KES, dates ISO text. All data synthetic.
CREATE TABLE property (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, county TEXT NOT NULL, area TEXT NOT NULL, category TEXT NOT NULL
);
CREATE TABLE unit (
    id INTEGER PRIMARY KEY, property_id INTEGER NOT NULL REFERENCES property(id), code TEXT NOT NULL,
    unit_type TEXT NOT NULL, bedrooms INTEGER NOT NULL, list_rent INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('OCCUPIED','VACANT')),
    UNIQUE (property_id, code)
);
CREATE TABLE tenant (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, phone TEXT NOT NULL, language TEXT NOT NULL,
    employment TEXT NOT NULL, joined_on TEXT NOT NULL
);
CREATE TABLE lease (
    id INTEGER PRIMARY KEY, unit_id INTEGER NOT NULL REFERENCES unit(id), tenant_id INTEGER NOT NULL REFERENCES tenant(id),
    start_date TEXT NOT NULL, end_date TEXT, rent INTEGER NOT NULL, deposit INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ACTIVE','ENDED'))
);
CREATE TABLE invoice (
    id INTEGER PRIMARY KEY, lease_id INTEGER NOT NULL REFERENCES lease(id), period_start TEXT NOT NULL,
    due_date TEXT NOT NULL, total INTEGER NOT NULL, amount_paid INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('UNPAID','PARTIAL','PAID'))
);
CREATE TABLE payment (
    id INTEGER PRIMARY KEY, lease_id INTEGER NOT NULL REFERENCES lease(id), invoice_id INTEGER REFERENCES invoice(id),
    paid_on TEXT NOT NULL, amount INTEGER NOT NULL, method TEXT NOT NULL CHECK (method IN ('MPESA','BANK','CASH')),
    reference TEXT NOT NULL
);
CREATE TABLE maintenance_request (
    id INTEGER PRIMARY KEY, unit_id INTEGER NOT NULL REFERENCES unit(id), tenant_id INTEGER REFERENCES tenant(id),
    category TEXT NOT NULL, description TEXT NOT NULL,
    priority TEXT NOT NULL CHECK (priority IN ('LOW','NORMAL','HIGH','EMERGENCY')),
    status TEXT NOT NULL CHECK (status IN ('OPEN','IN_PROGRESS','CLOSED')),
    created_at TEXT NOT NULL, closed_at TEXT, quote_amount INTEGER, safety_flag INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE expense (
    id INTEGER PRIMARY KEY, property_id INTEGER NOT NULL REFERENCES property(id), category TEXT NOT NULL,
    amount INTEGER NOT NULL, spent_on TEXT NOT NULL
);
CREATE INDEX ix_invoice_lease ON invoice(lease_id, period_start);
CREATE INDEX ix_payment_invoice ON payment(invoice_id);
CREATE INDEX ix_lease_unit ON lease(unit_id);

-- Views keep metric definitions in one place. :today is baked in at build time by generate.py.
CREATE VIEW v_invoice_state AS
SELECT i.id AS invoice_id, i.lease_id, l.tenant_id, l.unit_id, u.property_id, i.period_start, i.due_date,
       i.total, i.amount_paid, i.total - i.amount_paid AS balance, i.status,
       CASE WHEN i.total - i.amount_paid <= 0
            THEN (SELECT MAX(p.paid_on) FROM payment p WHERE p.invoice_id = i.id) END AS paid_in_full_on,
       CAST(julianday(COALESCE(
            CASE WHEN i.total - i.amount_paid <= 0
                 THEN (SELECT MAX(p.paid_on) FROM payment p WHERE p.invoice_id = i.id) END,
            '__TODAY__')) - julianday(i.due_date) AS INTEGER) AS days_late
FROM invoice i JOIN lease l ON l.id = i.lease_id JOIN unit u ON u.id = l.unit_id;

-- Overdue invoices with a balance, bucketed by days past due (docs/03 metric definitions).
CREATE VIEW v_arrears AS
SELECT *, CASE WHEN days_late <= 7 THEN '1-7' WHEN days_late <= 30 THEN '8-30'
               WHEN days_late <= 60 THEN '31-60' ELSE '60+' END AS bucket
FROM v_invoice_state WHERE balance > 0 AND due_date < '__TODAY__';
