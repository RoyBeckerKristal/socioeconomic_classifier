# socioeconomic_classifier
An ML-based classifier of socioeconomic index of a locality in Israel based on its ballot box results

This is a machine-learning based classification project.
It implements various classifiers and uses them to map numerical Israeli political data - distribution of kalpi (ballot box) votes in the Knesset elections to the socioeconomic index of the locality of the voters.

The original socioeconomic data are in excel files downloadable from the Israeli Central Bureau of Statistics.
The original voting data are in excel files downloadable from the Israeli Central Elections Committee.
Both required processing and merger, which I performed in excel and created the csv file serving as the input.

Following are the files:

* requirements.txt - the installation requirements

* socioeconomic_classifier.py - the main flow and various utilities

* sec_descriptive_stats.py - a script describing and charting the data and emerging tendencies

* sec_optimizers.py - optimization mechanisms for the various classifiers

* votes_knesset_25.csv - the data input file

* SEC - directory containing files storing classifier models already trained.

* Socioeconomic Classifier.pdf - a printout of a Google Slides presentation describing the project in detail.

 
To read more, please view the presentation.

For installation, download all the files to a directory, use the commandline (cmd), navigate to that directory, and enter:

pip install -r requirements.txt

To run, compile 'socioeconomic_classifier.py', and then, from the command line:

run_classification (input_file, output_file, explore_data, load_models, save_models, create_guesser)

input_file: string - name of file containing the input data. should be 'votes_knesset_25.csv'
output_file: string - name of file to contain the output, e.g. 'SEC_output.txt'
explore_data: boolean - True if you want to create charts describing the data.
load_models: boolean - True if you want to use pre-existing classifier models. False if you want to train new ones.
If True but models are absent, they are trained anyway.
save_models: boolean - True if you want classifier models to be saved to files.
create_guesser: boolean - True if you want the program to interact and be used to predict socioeconomic index per kalpi upon request.

After you enter "run_classification", the program will run, and, depending on what classifier it is requested to train if any, would dump all the output to the output file, including details of optimal hyperparameters for each classifier, and performance evaluation on the test data.

The input data contain 2363 kalpiot (ballot boxes) with SE-index, 500 out of which are randomly selected and withheld as test data. All classifier tests are based on them.
The input data file contains data also for the remaining ~9400 kalpiot, where there's no SE-index.
These data enable you to test the classifier on unclassified kalpiot.

For any inquiries please contact me via linkedin:
https://www.linkedin.com/in/roy-becker-kristal-71696634/

Roy Becker-Kristal
