System Requirements:

- Python, pip
- (Optional) [virtualenv](https://virtualenv.pypa.io/en/latest/)

To start the [Jupyter Notebook](https://jupyter.org/index.html):

```bash
# Clone the repo
git clone https://github.com/dennybritz/rnn-tutorial-rnnlm
cd rnn-tutorial-rnnlm

# Create a new virtual environment (optional, but recommended)
virtualenv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
# Start the notebook server
jupyter notebook