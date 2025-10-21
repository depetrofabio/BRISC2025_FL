FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Installa pacchetti base e Java
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        unzip \
        openjdk-21-jdk \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Find the correct Java home path dynamically
# This command gets the path to the installed Java and sets JAVA_HOME
RUN export JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:bin/java::") && \
    echo "export JAVA_HOME=$JAVA_HOME" >> /etc/profile.d/spark_env.sh && \
    echo "export PATH=$JAVA_HOME/bin:$PATH" >> /etc/profile.d/spark_env.sh

# The ENV for JAVA_HOME is now redundant since it's set dynamically
# ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
# ENV PATH=$JAVA_HOME/bin:$PATH

# Aggiorna pip
RUN pip install --upgrade pip

# Installa librerie Python per Data Science
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    jupyterlab \
    pyspark \
    torch \
    torchvision \
    torchaudio \
    nltk \
    wandb

WORKDIR /workspace

EXPOSE 8888

CMD ["bash"]