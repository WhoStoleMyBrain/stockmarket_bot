#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE simulation;
    GRANT ALL PRIVILEGES ON DATABASE simulation TO $POSTGRES_USER;
EOSQL
