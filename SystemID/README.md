# Question 2: #
This directory contains resources that were created to answer Question 2: Can we use machine learning to identify the control and stability derivitives of a small RC helicopter?

## Useful Files: ##
1. parameter_estimation_simulated_data_rev2.ipynb
2. quanser_2DOF_simulation.ipynb
3. simulated_outputs.csv
4. aero2_data.csv (all three of them; data generated in different runs)

## Comments: ##
### parameter_estimation_simulated_data_rev2.ipynb ###
- Most useful, working version of the code. Revision three contained some changes that didn't end up working.
- The code was written in interactive python notebooks (ipynb's) which allow you to execute various cells one at a time (ctrl+enter).
- ipynb's are useful for hands on developement of code and especially machine learning because you can make minor changes to some secions of the code without needing to rerun the whole thing which can be time consuming
- Workflow:
  - Load data
  - Preprocess data
    - Generate power spectral density to determine cutoff frequency
    - High pass butterworth filter at cutoff frequency
  - Setup ML model
    - Custom loss function using physics equation
    - Training loop
  - Predict using unseen data
  - Direct comparison of parameter results
  - Indirect comparison via simulation of dynamics

 ### quanser_2DOF_simulation.ipynb ###
- Generates synthetic data used to train the neural network
- Uses a state space model provided by Quanser and uses a Python implementation Matlab's lsim function to run a simulation on my own signal
- Applies noise to the simulation data to try and replicate measurement noise that might be encountered in a real experiment
- Checks the signal to noise ratio (SNR). Recommendation from Morelli and Klein is that it must be 10dB or higher to perform good analysis on so I've kept it in that threshold for my tests
- Saves synthetic data to a CSV file.
- This is still a workbook and it's pretty messy. Sorry I didn't have time to clean it up (I'm writing this readme during the summer ): )

### simulated_outputs.csv ###
- Data generated from the simulation that is run in 'quanser_2DoF_simulation.ipynb'

### aero2_data.csv ###
- A small amount of data that was gathered by actually running the Quanser Aero2
- There are controlled and uncontrolled versions of this data

#### Good luck and let me know if you have questions ####
- vanderwiel.gerrit@gmail.com
- gerritv@vt.edu
- (505) 412-0403
