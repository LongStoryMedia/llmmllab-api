-- Create extensions as superuser (postgres) before app user exists
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the app user and grant privileges
CREATE USER lsm WITH CREATEDB;
GRANT ALL PRIVILEGES ON DATABASE llmmllab TO lsm;
GRANT ALL ON SCHEMA public TO lsm;
