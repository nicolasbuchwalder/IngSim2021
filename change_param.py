import json 

with open('parameters.json') as fp:
	parameters = json.load(fp)

while(1):

	chosen_parameter = input(f"Current paramters:{list(parameters.keys())}\nWrite parameter to change or quit (type quit):")

	if chosen_parameter in parameters.keys():
		print(f"Current value of {chosen_parameter}:\t{parameters[chosen_parameter]}")
		new_value = input('Give new value: ') 
		parameters[chosen_parameter] = new_value
		print(f"Value of {chosen_parameter} changed to:\t{parameters[chosen_parameter]}")
	elif chosen_parameter == 'quit':
		print("---------")
		print(f"Final values: {parameters}")
		with open('parameters.json','w') as fp:
			json.dump(parameters,fp)
		break
	else:
		print("This parameter does not exist, please try again.")
	print("---------")