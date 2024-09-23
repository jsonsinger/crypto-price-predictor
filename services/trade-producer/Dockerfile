# Create a docker image for the trade-producer service

# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install Python Poetry 1.8.3
RUN pip install poetry==1.8.3

# Copy the source code to the working directory
COPY . /app

# Install the dependencies
RUN poetry install

# Run the container
CMD ["poetry", "run", "python", "-m", "src.main"]