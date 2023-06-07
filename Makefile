.PHONY:	
	clean
	coverage
	coverage-print

coverage:
	pytest --cov-report html --cov=turtlemd tests/

coverage-print:
	pytest --cov-report html --cov=turtlemd -rP tests/
 
clean:
	find -name \*.pyc -delete
	find -name \*.pyo -delete
	find -name __pycache__ -delete
	find -name \*.so -delete

