import smoothing
import matlab.engine
import clean_data

def main():
	IDs = [2,5,6,7]

	#smoothing.main(folder='Collected_11th_Feb')
	eng = matlab.engine.start_matlab()
	eng.digital_filter()
	clean_data.main(IDs,target_dir='Data/Processed_Data_1')

if __name__ == '__main__':
	main()