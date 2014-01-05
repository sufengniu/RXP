/*
 *  Copyright 2013 
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
 * The example shows following operations:
 *
 * 1) Including the header file PFAC.h which resides in directory $(PFAC_LIB_ROOT)/include.
 *    This header file is necessary because it contains declaration of APIs.
 *
 * 2) Initializing the PFAC library by creating a PFAC handle 
 *    (PFAC binds to a GPU context implicitly. If an user wants to bind a specific GPU, 
 *    he must call cudaSetDevice() explicitly before calling PFAC_create() ).
 *
 * 3) Reading patterns from a file and PFAC would create transition table both on the CPU side and the GPU side.
 *
 * 4) Dumping transition table to "table.txt", the content of table is shown in Figure 1 of user guide.
 *
 * 5) Reading an input stream from a file.
 *
 * 6) Performing matching process by calling the PFAC_matchFromHost() function.
 *
 * 7) Showing matched results.
 *
 * 8) Destroying the PFAC handle.
 *
 *
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <vector>
#include <limits.h>
#include <time.h>

#include <cuda_runtime.h>
#include <PFAC.h>

#define BILLION 1000000000L

#define Kilo	1000
#define Mega	1000*Kilo
#define Giga	1000*Mega

//#define	_MEASURE_

#define PROC_SIZE 700*Mega

void charcpy(char *target, char *source){
	while(*source)
	{
		*target = *source;
		source++;
		target++;
	}
	*target = '\0';
}

int main(int argc, char **argv)
{	
	char patternFile[] = "../test/pattern/space_pattern" ;
	PFAC_handle_t handle ;
	PFAC_status_t PFAC_status ;
	size_t input_size ;    
	char *h_inputString = NULL ;
	int *h_matched_result = NULL ;

	struct timespec start, stop;
	double load_accum, comp_accum, printf_accum;

#ifdef _MEASURE_
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
	printf("max string size should less than %d\n", LINE_MAX);
#endif

	// step 1: create PFAC handle 
	PFAC_status = PFAC_create( &handle ) ;
	assert( PFAC_STATUS_SUCCESS == PFAC_status );

	// step 2: read patterns and dump transition table
	PFAC_status = PFAC_readPatternFromFile( handle, patternFile) ;
	if ( PFAC_STATUS_SUCCESS != PFAC_status ){
		printf("Error: fails to read pattern from file, %s\n", PFAC_getErrorString(PFAC_status) );
		exit(1) ;
	}

	// step 3: prepare input string
	char *inputString;
	int offset = 0;
	size_t len = 0;
	inputString = (char *)malloc(sizeof(char)*PROC_SIZE);	
//	while(fgets(h_inputString, LINE_MAX, stdin) != NULL) {	
	while(getline(&h_inputString, &len, stdin) > 0) {

		h_inputString[strlen(h_inputString)-1] = ' ';	// replace each \n as blank
		charcpy(inputString+offset, h_inputString);
		offset += strlen(h_inputString);
	}

	input_size = strlen(inputString);
	h_matched_result = (int *) malloc (sizeof(int)*input_size);	

	memset (h_matched_result, 0, sizeof(int)*input_size);	

#ifdef _MEASURE_
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &stop);
	load_accum = (stop.tv_sec - start.tv_sec)+(double)(stop.tv_nsec-start.tv_nsec)/(double)BILLION;

	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);	
#endif
	// step 4: run PFAC on GPU           
	PFAC_status = PFAC_matchFromHost( handle, inputString, input_size, h_matched_result ) ;
	if ( PFAC_STATUS_SUCCESS != PFAC_status ){
		printf("Error: fails to PFAC_matchFromHost, %s\n", PFAC_getErrorString(PFAC_status) );
		exit(1) ;	
	}     

#ifdef _MEASURE_
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &stop);
	comp_accum = (stop.tv_sec - start.tv_sec)+(double)(stop.tv_nsec-start.tv_nsec)/(double)BILLION;

	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
#endif
	// step 5: output matched result
	int i, count=0;
	for (int i = 0; i < input_size; i++) {
		if (h_matched_result[i] != 0) {
			count++;
		}	
	}
	
	char *res = NULL;
	res = (char *)malloc(input_size+3*count*sizeof(char));	

	char *res_tmp = res;
	for (i = 0; i < input_size; i++) {
	
		if (h_matched_result[i] == 0) {
			*res = inputString[i];
			res++;
		} else {
			if ((i > 0) && ( h_matched_result[i-1] == 0)) {
				strcpy(res, "\t1\n");
				res += 3;
			} else {
				continue;
			}
		}	

	/*	if (h_matched_result[i] != 0) {
			
			if ((i > 0) && ( h_matched_result[i-1] == 0)) {
				strcpy(res, "\t1\n");	
				res += 3;
			} else {
				continue;
			}
		}
		else {	 
			
			*res = inputString[i];
			res++;	
		}*/
	}

	res = res_tmp;

	printf("%s\n", res);
	//fwrite(res, 1, strlen(res), stdout);

#ifdef _MEASURE_

	// parse in parallel

	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &stop);
	printf_accum = (stop.tv_sec - start.tv_sec)+(double)(stop.tv_nsec-start.tv_nsec)/(double)BILLION;

	printf("data load done in %lf second\n", load_accum);
	printf("computation done in %lf second\n", comp_accum);	
	printf("printf done in %lf second\n", printf_accum);	
	float throughput = ((float)input_size*8.0)/(comp_accum*1000000000);
	printf("throughput is %lf Gbps\n", throughput);

#endif

	PFAC_status = PFAC_destroy( handle ) ;
	assert( PFAC_STATUS_SUCCESS == PFAC_status );
	
	free(res);
	free(inputString);
	free(h_inputString);
	free(h_matched_result); 

	return 0;
}


