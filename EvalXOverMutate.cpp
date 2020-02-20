#include <iostream>  // cout
#include <fstream>   // ifstream
#include <string.h>  // strncpy
#include <stdlib.h>  // rand
#include <math.h>    // sqrt, pow
#include <omp.h>     // OpenMP
#include <algorithm> //sort
#include "Timer.h"
#include "Trip.h"

using namespace std;

/*
 * Method to calculate distance between two given cities
 */
float calc_dist(char city1, char city2, int coordinates[CITIES][2]){
	int indexA = ( city1 >= 'A' ) ? city1 - 'A' : city1 - '0' + 26;
	int indexB = ( city2 >= 'A' ) ? city2 - 'A' : city2 - '0' + 26;
	float distance = sqrt(pow((coordinates[indexA][0]-coordinates[indexB][0]),2)+pow((coordinates[indexA][1]-coordinates[indexB][1]),2));
	return distance;
}

/*
 * Operator overloading to enable sorting of array of Trip objects
 */
bool operator<(Trip const & a, Trip const & b)
{
    return a.fitness < b.fitness;
}

/*
 * Method to evaluate total distance of each trip in the population and to sort them in ascending order
 */
void evaluate( Trip trip[CHROMOSOMES], int coordinates[CITIES][2] ){
	#pragma omp parallel for default( none ) shared( coordinates,trip )
	for(int i=0;i<CHROMOSOMES;i++){
		float total_distance = 0;
		Trip chromosome = trip[i];
		//evaluate individual trips
//		#pragma omp parallel for default( none ) shared( chromosome,coordinates ) reduction( +:total_distance )
		for(int j=0;j<CITIES-1;j++){
			int indexA = ( chromosome.itinerary[j] >= 'A' ) ? chromosome.itinerary[j] - 'A' : chromosome.itinerary[j] - '0' + 26;
			int indexB = ( chromosome.itinerary[j+1] >= 'A' ) ? chromosome.itinerary[j+1] - 'A' : chromosome.itinerary[j+1] - '0' + 26;
			float distance = sqrt(pow((coordinates[indexA][0]-coordinates[indexB][0]),2)+pow((coordinates[indexA][1]-coordinates[indexB][1]),2));
			total_distance +=distance;
		}
		trip[i].fitness=total_distance;
	}
	//sort
	sort(trip, trip+CHROMOSOMES);
}

/*
 * Returns a complimentary city name, one of: ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
 * such that complimentary city of A is 9, B is 8 and so on
 */
char getComplimentaryCity(char val ) {
  int n = ( val >= 'A' ) ? val - 'A' : val - '0' + 26;
  char city = abs(n-CITIES+1);
  if ( city < 26 )
    city += 'A';
  else
    city = city - 26 + '0';
  return city;
}

/*
 * Creates a complimentary trip by replacing each city of a given trip with its complimentary city
 */
void complementary_child(char arr[CITIES],char complimentary_arr[CITIES]){
	#pragma omp parallel for default( none ) shared( complimentary_arr,arr )
	for(int i=0;i<CITIES;i++){
		complimentary_arr[i] =getComplimentaryCity(arr[i]);
	}
}

/*
 * Method to find if a city is already visited in a trip
 */
bool find_in_array_parallel(char arr[CITIES],char val){
	bool flag = false;
	#pragma omp parallel for default( none ) shared( arr,flag,val)
	for(int i=0;i<CITIES;i++){
		if(arr[i]==val){
			flag  = true;
		}
	}
	return flag;
}

/*
 * Method that returns the position of a city if it's already visited in a trip
 */
int find_pos_in_array_parallel(char arr[CITIES],char val){
	int pos = CITIES;
	#pragma omp parallel for default( none ) shared( arr,pos,val)
	for(int i=0;i<CITIES;i++){
		if(arr[i]==val){
			pos = i;
		}
	}
	return pos;
}

/*
 * Method to crossover parent i and parent i+1 to spawn two children child i and child i+1
 * where child i+1 is complimentary trip of child i
 */
void crossover( Trip parents[TOP_X], Trip offsprings[TOP_X], int coordinates[CITIES][2] ){
//	cout<<"in crossover\n";
	#pragma omp parallel for default( none ) shared( coordinates,offsprings,parents)
	for(int i=0;i<TOP_X-1;i+=2){
		Trip parent1 = parents[i];
		Trip parent2 = parents[i+1];
		memset(offsprings[i].itinerary,'\0',CITIES);
//		cout<<"parent1 "<<parent1.itinerary<<"\t"<<"parent2 "<<parent2.itinerary<<"\n";
		offsprings[i].itinerary[0]=parent1.itinerary[0];
		for(int j=1;j<CITIES;j++){
			//position of previous city in offspring in parent1
			int pos1 = find_pos_in_array_parallel(parent1.itinerary,offsprings[i].itinerary[j-1]);
			char city1 = (pos1<CITIES-1)?parent1.itinerary[pos1+1]:parent1.itinerary[0];
			int pos2 = find_pos_in_array_parallel(parent2.itinerary,offsprings[i].itinerary[j-1]);
			char city2 = (pos2<CITIES-1)?parent2.itinerary[pos2+1]:parent2.itinerary[0];
			bool city1_present = find_in_array_parallel(offsprings[i].itinerary,city1);
			bool city2_present = find_in_array_parallel(offsprings[i].itinerary,city2);
			if(city1_present && city2_present){
				int rand_pos= (pos1<CITIES-2)?pos1+2:0;
//				srand(time(0));
//				int rand_pos = rand()%CITIES;
				for(int k=0;k<CITIES;k++){
					if(!find_in_array_parallel(offsprings[i].itinerary,parent1.itinerary[rand_pos])){
						offsprings[i].itinerary[j]=parent1.itinerary[rand_pos];
						break;
					}else if(!find_in_array_parallel(offsprings[i].itinerary,parent2.itinerary[rand_pos])){
						offsprings[i].itinerary[j]=parent2.itinerary[rand_pos];
						break;
					}else{
						if(rand_pos < CITIES-1){
							rand_pos++;
						}else{
							rand_pos=0;
						}
					}
				}
			}else if(city1_present){
				offsprings[i].itinerary[j]=city2;
			}else if(city2_present){
				offsprings[i].itinerary[j]=city1;
			}else{
				float distance1 = calc_dist(offsprings[i].itinerary[j-1],city1,coordinates);
				float distance2 = calc_dist(offsprings[i].itinerary[j-1],city2,coordinates);

				if(distance1 < distance2){
					offsprings[i].itinerary[j]=city1;
				}else{
					offsprings[i].itinerary[j]=city2;
				}
			}

		}
		complementary_child(offsprings[i].itinerary,offsprings[i+1].itinerary);
	}
}

/*
 * Method to swap two cities in a trip selected in random for a given probability
 */
void mutate( Trip offsprings[TOP_X] ){
	srand(time(0));
	for(int i=0;i<TOP_X;i++){
		if(rand()%100<MUTATE_RATE){
			int pos1 = rand() % CITIES;
			int pos2 = rand() % CITIES;
			char city = offsprings[i].itinerary[pos1];
			offsprings[i].itinerary[pos1] = offsprings[i].itinerary[pos2];
			offsprings[i].itinerary[pos2] = city;
		}
	}
}
