FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
WORKDIR /repos/tts_project

# Install requirements for torchaudio
RUN pip install sox && conda install torchaudio==0.11.0 -c pytorch && conda install -c conda-forge librosa

# Install requirements
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Install wget
RUN apt-get update ; apt-get install unzip ; apt-get install tar

# Copy the contents of repository
COPY . .

# Expose port
EXPOSE 3000