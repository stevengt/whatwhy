
ssh -i ~/Downloads/whatwhy-ec2-key-pair.pem ubuntu@ec2-3-227-18-34.compute-1.amazonaws.com


echo "Memory available in KB"
grep MemTotal /proc/meminfo | awk '{print $2}' 

echo "Num CPU cores available"
grep 'cpu cores' /proc/cpuinfo | awk '{print $4}' 


sudo apt-get install -y awscli git swig3.0
aws configure

# conda create -n whatwhy python=3.6
# conda activate whatwhy
conda update --all

source activate tensorflow_p36
python -c "import tensorflow as tf; hello = tf.constant('Hello, TensorFlow!'); sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)); print(sess.run(hello))"

cd /home/ubuntu
git clone https://github.com/stevengt/whatwhy.git
cd whatwhy
# Remove tensorflow from setup.py
pip install .

mkdir -p /home/ubuntu/whatwhy-data
mkdir -p /home/ubuntu/whatwhy-data/tf-model

cd /home/ubuntu/whatwhy-data
aws s3 cp s3://whatwhy-data/News-Articles/all-the-news/vectorizers ./vectorizers/ --recursive

echo "Run this command in Python, and backup the model before exiting if something goes wrong."
echo "import whatwhy.data_analysis.main"
python

aws s3 cp /home/ubuntu/whatwhy-data/tf-model s3://whatwhy-data/News-Articles/all-the-news/tf-model/ --recursive
