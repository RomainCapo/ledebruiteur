import unittest
import os
import subprocess

class TestDenoiseMethods(unittest.TestCase):

	SUCCESS_RET_CODE = 0
	BASE_PATH = "test_image"
	FILTERS = {
			'gaussian_substract': 0,
			'laplacian_filter': 1,
			'no_filter': 2,
			'wiener_filter': 3,
			'high_pass_filter': 4,
		}

	def test_denoiser(self):
		FILE_IN = os.path.join(self.BASE_PATH, 'img96_UniformNoise.jpg')

		return_codes = []
		for filter_name, filter_opt in self.FILTERS.items():
			file_out = os.path.join(self.BASE_PATH, filter_name + ".jpg")
			sp = subprocess.Popen(['python', 'denoise.py', "-i", FILE_IN, "-o", file_out , "-p", f'{filter_opt}'], 
								  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
				
			out, err = sp.communicate()
			print(f'{out}')
			print(f'{err}')
			
			return_codes.append(sp.returncode)

		expected_status_code = [self.SUCCESS_RET_CODE for i in range(0, len(self.FILTERS.keys()))]
		self.assertEqual(return_codes, expected_status_code)
	
	
	def tearDown(self):
		output_files = list(map(lambda name: os.path.join(self.BASE_PATH, name + ".jpg"), self.FILTERS.keys()))
		for output_file in output_files:
			os.remove(output_file)

if __name__ == '__main__':
    unittest.main()
