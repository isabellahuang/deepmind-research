import matplotlib.pyplot as plt
import os
import csv
import argparse
import math

# Create command line flag options
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--checkpoint_dir', default='None', type=str)
args = parser.parse_args()


# initialize figures
fig, (ax1, ax2) = plt.subplots(2)
# fig, ax2 = plt.subplots(1)
plt.figure(3)


folders = args.checkpoint_dir.split(",")
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
markers = ['--', '.']
for f_ind, folder in enumerate(folders):
	folder_path = os.path.join('meshgraphnets/data/chk', folder)
	loss_file = [k for k in os.listdir(folder_path) if "losses.txt" in k][0]
	loss_file = os.path.join(folder_path, loss_file)


	train_losses = []
	test_losses = []
	test_pos_mean_errors, test_pos_final_errors = [], []
	test_stress_mean_errors, test_stress_final_errors = [], []
	baseline_pos_mean_error, baseline_pos_final_error = None, None
	baseline_stress_mean_error, baseline_stress_final_error = None, None

	with open(loss_file, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')

		for row in reader:
			train_losses.append(float(row[1]))
			test_losses.append(float(row[2]))
			test_pos_mean_errors.append(float(row[3]))
			test_pos_final_errors.append(float(row[4]))

			if len(row) >= 8:
				baseline_pos_mean_error = float(row[5])
				baseline_pos_final_error = float(row[6])
				test_stress_mean_errors.append(float(row[7]))
				test_stress_final_errors.append(float(row[8]))
				baseline_stress_mean_error = float(row[9])
				baseline_stress_final_error = float(row[10])


	# ax1.plot(train_losses, label="Train loss " + folder)
	ax1.plot(test_losses, label="Test loss " + folder)
	ax1.legend()
	K = 150
	ax2.plot(train_losses[:], label="Train loss " + folder, color=colors[f_ind])
	ax2.set_title("Train losses")
	ax2.legend()

	plt.figure(3)
	plt.plot(test_pos_mean_errors[:], alpha=0.7, label="Test mean loss " + folder, color=colors[f_ind])
	plt.plot(test_pos_final_errors[:], '--', alpha=0.7, label="Test final loss " + folder, color=colors[f_ind])
	plt.ylabel("Error [m]")
	if baseline_pos_mean_error and baseline_pos_final_error:
		plt.plot([baseline_pos_mean_error] * len(test_pos_mean_errors), color='gray')
		plt.plot([baseline_pos_final_error] * len(test_pos_final_errors), '--', color='gray')
	plt.title("Pos test losses")
	plt.legend()

	# '''
	plt.figure(4)
	plt.plot(test_stress_mean_errors[:], alpha=0.7, label="Test mean loss " + folder, color=colors[f_ind])
	plt.plot(test_stress_final_errors[:], '--', alpha=0.7, label="Test final loss " + folder, color=colors[f_ind])
	plt.ylabel("Error [Pa]")
	if baseline_stress_mean_error and baseline_stress_final_error:
		plt.plot([baseline_stress_mean_error] * len(test_stress_mean_errors), color='gray')
		plt.plot([baseline_stress_final_error] * len(test_stress_final_errors), '--', color='gray')
	plt.title("Stress test losses")
	plt.legend()
	# '''

plt.show()
