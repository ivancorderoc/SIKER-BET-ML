FROM python:3.10

WORKDIR /app

# Copiar solo requirements primero para aprovechar cache
COPY requirements.txt .

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y build-essential

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código y recursos necesarios
COPY . .

EXPOSE 8000

CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
