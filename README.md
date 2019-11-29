# Airbnb Recommendations

## Set up
Download the dataset and store it within the project dir as `airbnb-recruiting-new-user-bookings`.
Next create the dataseet. Here we will create our engineered features and process the dataset.
```bash
cd data
python make_dataset.py
```
Doing this, you'll output a processed baseline csv file `airbnb-recruiting-new-user-bookings/train_users_2-processed.csv`

## Train and evaluate model
```bash
cd <PROJECT_DIR>
python train_model.py
```

Directory Structure
------------

The directory structure of the Airbnb Recommendations project looks like this: 

```
├── README.md                               <- The top-level README for developers using this project.
├── data                                    <- Scripts to download or generate data
│   ├── dataloader.py
│   ├── d_utils.py
│   └── prepare_db.py
│
├── models                                  <- Scripts to evaluate model
│   └── eval_model.py
├── airbnb-recruiting-new-user-bookings     <- Expected data file
│
└── train_model.py                          <- Main script
```
