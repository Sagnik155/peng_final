# 1. Use an official PyTorch runtime parent image (CUDA 12.4 compatible)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# 2. Set environment variables to prevent Python from buffering stdout
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 3. Create a non-root user 'algorithm' (Strict Grand Challenge requirement)
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

# 4. Create the required strict input/output directories and set ownership
RUN mkdir -p /opt/app /input /output \
    && chown -R algorithm:algorithm /opt/app /input /output

# 5. Set the working directory
WORKDIR /opt/app

# 6. Copy requirements and install dependencies
# Doing this before copying the code leverages Docker caching
COPY --chown=algorithm:algorithm requirements.txt /opt/app/
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy your inference method and predict script
COPY --chown=algorithm:algorithm method/ /opt/app/method/
COPY --chown=algorithm:algorithm predict.py /opt/app/

# 8. Switch to the non-root user before execution
USER algorithm

# 9. Define the default command to run your prediction script
ENTRYPOINT ["python", "predict.py"]