import psycopg2
import psycopg2.extras
from datetime import datetime
import os

class DatabaseManager:
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'postgres'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'housing_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password')
        }
    
    def get_connection(self):
        return psycopg2.connect(**self.connection_params)
    
    def save_prediction(self, prediction_data):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            insert_query = '''
            INSERT INTO predictions (
                surface, rooms, bedrooms, age, location_score,
                has_garden, has_parking, has_balcony, energy_class,
                predicted_price, prediction_date
            ) VALUES (
                %(surface)s, %(rooms)s, %(bedrooms)s, %(age)s, %(location_score)s,
                %(has_garden)s, %(has_parking)s, %(has_balcony)s, %(energy_class)s,
                %(predicted_price)s, NOW()
            )
            '''
            
            cursor.execute(insert_query, prediction_data)
            conn.commit()
        finally:
            conn.close()
    
    def save_training_data(self, training_data):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Vider la table existante
            cursor.execute("DELETE FROM training_data")
            
            insert_query = '''
            INSERT INTO training_data (
                surface, rooms, bedrooms, age, location_score,
                has_garden, has_parking, has_balcony, energy_class, price
            ) VALUES (
                %(surface)s, %(rooms)s, %(bedrooms)s, %(age)s, %(location_score)s,
                %(has_garden)s, %(has_parking)s, %(has_balcony)s, %(energy_class)s, %(price)s
            )
            '''
            
            cursor.executemany(insert_query, training_data)
            conn.commit()
        finally:
            conn.close()
    
    def get_training_data(self):
        conn = self.get_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT * FROM training_data")
            return cursor.fetchall()
        finally:
            conn.close()
    
    def get_predictions(self):
        conn = self.get_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT * FROM predictions ORDER BY prediction_date DESC LIMIT 100")
            return cursor.fetchall()
        finally:
            conn.close()