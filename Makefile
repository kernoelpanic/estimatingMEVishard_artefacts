.PHONY: html

HTMLPATH?=$(pwd)./notebooks/

# initialize virtual environment in local folder
# same as:
# $ virtualenv -p /usr/local/bin/python3 venv3.37
# $ source venv3/bin/activate
# $ python3 -m pip install -r requirements.txt
init:
	( \
		virtualenv -p /usr/bin/python3 venv3; \
  	. ./venv3/bin/activate; \
		python --version; \
		python3 -m pip install -r requirements.txt; \
	)

# install missing packages newly added to requirements.txt
install:
	( \
    . ./venv3/bin/activate; \
    python --version; \
		python3 -m pip install --upgrade pip; \
    python3 -m pip install -r requirements.txt; \
  )

# start jupyter notebook in virtual environment 
start:
	( \
		. venv3/bin/activate; \
		jupyter nbextension enable --py --sys-prefix qgrid; \
		jupyter nbextension enable --py --sys-prefix widgetsnbextension; \
		jupyter notebook \
	)

# Use as follows:
# $ make html HTMLPATH=./value_vs_rewards/
html:
	( \
    . ./venv3/bin/activate; \
		jupyter nbconvert --to html $(HTMLPATH)*.ipynb \
	) 

# Extract source code form notebook to python file
# $ make extract
extract:
	( \
	 . ./venv3/bin/activate; \
	NBFILE="./notebooks/games_profit.ipynb" PYFILE="./src/games/multigames.py" python3 ./src/util/nb_extract.py \
	)	 

clean:
	-rm -i $(HTMLPATH)*.html

