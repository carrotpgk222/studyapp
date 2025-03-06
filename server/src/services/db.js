const sqlite3 = require('sqlite3').verbose();
const { query } = require('express');
const path = require('path');

// Adjust path if your .db file is in a different location
const dbPath = path.join(__dirname, '../../../study_app.db');

// Connect to SQLite database
const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error('Failed to connect to SQLite database:', err);
  } else {
    console.log('Connected to SQLite database');
  }
});

const pool = {
  query: function (sql, params, callback) {
    if (sql.trim().toUpperCase().startsWith("SELECT")) {
      // Use db.all() for SELECT statements
      db.all(sql, params, (err, rows) => {
        if (callback) callback(err, rows);
      });
    } else {
      // Use db.run() for INSERT, UPDATE, DELETE
      db.run(sql, params, function (err) {
        if (callback) {
          callback(err, { lastID: this.lastID, changes: this.changes });
        }
      });
    }
  }
};

module.exports = { pool, db };
