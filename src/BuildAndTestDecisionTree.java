import java.util.*;
import java.io.*;

public class BuildAndTestDecisionTree
{
  // "Main" reads in the names of the files we want to use, then reads 
  // in their examples.
  public static void main(String[] args)
  {   
    if (args.length != 2)
    {
      System.err.println("You must call BuildAndTestDecisionTree as " + 
			 "follows:\n\njava BuildAndTestDecisionTree " + 
			 "<trainsetFilename> <testsetFilename>\n");
      System.exit(1);
    }    

    // Read in the file names.
    String trainset = args[0];
    String testset  = args[1];

    // Read in the examples from the files.
    ListOfExamples trainExamples = new ListOfExamples();
    ListOfExamples testExamples  = new ListOfExamples();
    if (!trainExamples.ReadInExamplesFromFile(trainset) ||
        !testExamples.ReadInExamplesFromFile(testset))
    {
      System.err.println("Something went wrong reading the datasets ... " +
			   "giving up.");
      System.exit(1);
    }
    else
    {

      trainExamples.DescribeDataset();
      testExamples.DescribeDataset();
      //trainExamples.PrintThisExample(0);  // Print out an example
      //trainExamples.PrintAllExamples(); // Don't waste paper printing all 
                                          // of this out!
      //testExamples.PrintAllExamples();  // Instead, just view it on the screen
      
      //double totalEntropy = totalEntropy(trainExamples);
      TreeNode root = BuildTreeAndReturnRoot(trainExamples);
      
      Hashtable<Example, String> predictions = TestTree(root, testExamples);
      
      	double numberOfExamples = testExamples.numExamples;
		double numberOfCorrectPredictions = 0;
		int numberOfIncorrectPredictions = 0;
		int i = 0;
		ArrayList<Example> incorrectlyPredictedExamples = new ArrayList<Example>();
		String predictedLabel = "";
		String actualLabel = "";
		String printString = "";
		Example testExample;
		Example missedExample;
	    Enumeration<Example> enumeration = predictions.keys();
	   
		System.out.println();
		
	    //iterate through Hashtable keys
	    while (enumeration.hasMoreElements()) {
	    	
	    	testExample = enumeration.nextElement();
	    	predictedLabel = predictions.get(testExample);
	    	if (predictedLabel.equals(testExample.label)) {
	    		numberOfCorrectPredictions++;
	    	}
	    	
	    	else if (!(predictedLabel.equals(testExample.label))) {
	    		numberOfIncorrectPredictions++;
	    		incorrectlyPredictedExamples.add(testExample);
	    	}
	    }
	    
		System.out.println();
		
		double accuracy = 100 * (numberOfCorrectPredictions / numberOfExamples);
		System.out.println("Accuracy: " + accuracy + "% (" + (int) numberOfCorrectPredictions + "/" + (int) numberOfExamples + ")");
		
		System.out.println();
		
		System.out.println("Number of examples classified incorrectly: " + numberOfIncorrectPredictions);
		for (i = 0; i < numberOfIncorrectPredictions; i++) {
			missedExample = incorrectlyPredictedExamples.get(i);
			predictedLabel = "(predicted label = " + predictions.get(missedExample) + ", ";
			actualLabel = "actual label = " + missedExample.label + ")";
			printString = (i + 1) + ". " + missedExample.name + predictedLabel + actualLabel;
			System.out.println(printString);
		}
		
		PrintTree(0, root);
		//trainExamples.PrintAllExamples(); // Don't waste paper printing all of this out!
		//testExamples.PrintAllExamples();  // Instead, just look at it on the screen.
    }

    Utilities.waitHere("Hit <enter> when ready to exit.");
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  
	//RAS - This method takes in a ListOfExamples, creates a shell tree root node and empty path, and
  	//then calls the tree-building method and returns the generated tree's root node.
	public static TreeNode BuildTreeAndReturnRoot(ListOfExamples trainExamples) {
	
	List<BinaryFeature> path = new ArrayList<BinaryFeature>();
	TreeNode root = new TreeNode();

	buildTree(trainExamples, root, path, trainExamples);
	return root;
}

	//RAS - This method takes a ListOfExamples and returns the total entropy, or information needed.
	public static double totalEntropy(ListOfExamples trainExamples){
		
		Iterator<Example> iterator = trainExamples.iterator();
		Example example;
		String firstLabelValue = trainExamples.outputLabel.firstValue;
		String secondLabelValue = trainExamples.outputLabel.secondValue;
		double numberOfExamplesWithFirstOutputLabel = 0;
		double numberOfExamplesWithSecondOutputLabel = 0;
		int numberOfExamples = trainExamples.numExamples;
		
		while (iterator.hasNext()){
			
			example = iterator.next();
			if (example.label.equals(firstLabelValue)) {
				numberOfExamplesWithFirstOutputLabel++;
			}
			
			else if (example.label.equals(secondLabelValue)) {
				numberOfExamplesWithSecondOutputLabel++;
			}
		}
		
		double portionOfExamplesWithFirstOutputLabel = numberOfExamplesWithFirstOutputLabel / numberOfExamples;
		double portionOfExamplesWithSecondOutputLabel = numberOfExamplesWithSecondOutputLabel / numberOfExamples;
		
		return InfoNeeded(portionOfExamplesWithFirstOutputLabel, portionOfExamplesWithSecondOutputLabel);
	}

	//RAS - This method takes in a ListOfExamples and a (possibly empty) path and returns the index of the
	//ListOfExample's best feature (best meaning that it gives us the most information).
	public static int getBestFeatureIndex(ListOfExamples trainExamples, List<BinaryFeature> path){
		
		List<BinaryFeature> pathToUse;
		BinaryFeature feature = null;
		BinaryFeature bestFeature = null;
		double featureRemainder;
		double bestFeatureRemainder = 1;
		int bestFeatureIndex = 0;
		int numberOfFeatures = path.size();
		
		if (numberOfFeatures == 0) {
			pathToUse = new ArrayList<BinaryFeature>(Arrays.asList(trainExamples.features));
			numberOfFeatures = pathToUse.size();
		}
		
		else {
			pathToUse = copyPath(path);
		}
		
		for (int i = 0; i < numberOfFeatures; i++){
			
			feature = pathToUse.get(i);
			
			featureRemainder = RemainderOfFeature(trainExamples, feature);
			
			if (featureRemainder == 0) {
				bestFeatureIndex = FeatureIndex(trainExamples, feature.name);
				return bestFeatureIndex;
			}
			
 			if (featureRemainder < bestFeatureRemainder) {
				bestFeatureRemainder = featureRemainder;
				bestFeature = feature;
			}
		}
		
		if (bestFeature == null) {
			bestFeature = feature;
		}
		
		try {
			bestFeatureIndex = FeatureIndex(trainExamples, bestFeature.name);
		} catch (NullPointerException e) {
			System.err.println("NullPointerException: " + e.getMessage());
		}
		
		return bestFeatureIndex;
	}
	
			
	//RAS - This method takes in a ListOfExamples and a BinaryFeature and returns the remainder of the
	//feature per the definition of remainder in ID3.
	private static double RemainderOfFeature(ListOfExamples trainExamples, BinaryFeature feature) {
		
		Example example;
		Iterator<Example> iter = trainExamples.iterator();
		double numberOfExamplesWithFirstValue = 0;
		double numberOfExamplesWithSecondValue = 0;
		double numberOfExamplesWithFirstValueAndFirstLabelValue = 0;
		double numberOfExamplesWithFirstValueAndSecondLabelValue = 0;
		double numberOfExamplesWithSecondValueAndFirstLabelValue = 0;
		double numberOfExamplesWithSecondValueAndSecondLabelValue = 0;
		double numberOfExamples = trainExamples.size();
		double returnValue;
		int featureIndex = FeatureIndex(trainExamples, feature.name);
		boolean exampleHasFirstFeatureValue;
		boolean exampleHasSecondFeatureValue;
		String exampleFeatureValue;
		String firstValue = feature.firstValue;
		String secondValue = feature.secondValue;
		String firstLabelValue = trainExamples.outputLabel.firstValue;
		String SecondLabelValue = trainExamples.outputLabel.secondValue;
		String exampleLabel;
		
		while (iter.hasNext()){
			
			example = iter.next();
			exampleFeatureValue = example.get(featureIndex);
			exampleHasFirstFeatureValue = exampleFeatureValue.equals(firstValue);
			exampleHasSecondFeatureValue = exampleFeatureValue.equals(secondValue);
			exampleLabel = example.label;
			
			if (exampleHasFirstFeatureValue){
				numberOfExamplesWithFirstValue++;
			}
			
			if (exampleHasSecondFeatureValue){
				numberOfExamplesWithSecondValue++;
			}
			
			if (exampleHasFirstFeatureValue && exampleLabel.equals(firstLabelValue)){
				numberOfExamplesWithFirstValueAndFirstLabelValue++;
			}
			
			if (exampleHasFirstFeatureValue && exampleLabel.equals(SecondLabelValue)){
				numberOfExamplesWithFirstValueAndSecondLabelValue++;
			}
			
			if (exampleHasSecondFeatureValue && exampleLabel.equals(firstLabelValue)){
				numberOfExamplesWithSecondValueAndFirstLabelValue++;
			}
			
			if (exampleHasSecondFeatureValue && exampleLabel.equals(SecondLabelValue)){
				numberOfExamplesWithSecondValueAndSecondLabelValue++;
			}
		}
		
		double portionOfExamplesWithFirstValue = 0;
		double portionOfExamplesWithSecondValue = 0;
		double portionOfSecondValueExamplesWithFirstLabelValue = 0;
		double portionOfSecondValueExamplesWithSecondLabelValue = 0;
		
		if (!(numberOfExamples == 0)) {
			portionOfExamplesWithFirstValue = numberOfExamplesWithFirstValue / numberOfExamples;
			portionOfExamplesWithSecondValue = numberOfExamplesWithSecondValue / numberOfExamples;
		}

		double portionOfFirstValueExamplesWithFirstLabelValue = 0;
		double portionOfFirstValueExamplesWithSecondLabelValue = 0;
		
		if (!(numberOfExamplesWithFirstValue == 0)) {
			portionOfFirstValueExamplesWithFirstLabelValue = numberOfExamplesWithFirstValueAndFirstLabelValue / numberOfExamplesWithFirstValue;
			portionOfFirstValueExamplesWithSecondLabelValue = numberOfExamplesWithFirstValueAndSecondLabelValue / numberOfExamplesWithFirstValue;
		}

		if (!(numberOfExamplesWithSecondValue == 0)) {
		portionOfSecondValueExamplesWithFirstLabelValue = numberOfExamplesWithSecondValueAndFirstLabelValue / numberOfExamplesWithSecondValue;
		portionOfSecondValueExamplesWithSecondLabelValue = numberOfExamplesWithSecondValueAndSecondLabelValue / numberOfExamplesWithSecondValue;
		}
		
		double infoNeededPart1 = InfoNeeded(portionOfFirstValueExamplesWithFirstLabelValue, portionOfFirstValueExamplesWithSecondLabelValue);
		double returnValuePart1 = portionOfExamplesWithFirstValue * infoNeededPart1;
		
		double infoNeededPart2 = InfoNeeded(portionOfSecondValueExamplesWithFirstLabelValue, portionOfSecondValueExamplesWithSecondLabelValue);
		double returnValuePart2 = portionOfExamplesWithSecondValue * infoNeededPart2;
		
		returnValue = returnValuePart1 + returnValuePart2;
		
		return returnValue;
	}
	
	//RAS - This method is our implementation of the InfoNeeded method in ID3. It takes two doubles, which
	//add up to 1, and returns the output of the InfoNeeded method applied to these two doubles.
	private static double InfoNeeded(double portionWithFirstLabelValue, double portionWithSecondLabelValue) {
		
		double returnValue;
		boolean completeInformation = (portionWithFirstLabelValue == 0 || portionWithFirstLabelValue == 1);
		completeInformation = (completeInformation || (portionWithSecondLabelValue == 0 || portionWithSecondLabelValue == 1));
		
		if (completeInformation){
			returnValue = 0;
		}
		
		else {
			returnValue = - (portionWithFirstLabelValue) * (logBase2(portionWithFirstLabelValue))
					- (portionWithSecondLabelValue) * (logBase2(portionWithSecondLabelValue));
		}
		
		return returnValue;
	}

    //RAS - This method takes a ListOfExamples as input and returns a copy of the ListOfExamples.
	public static ListOfExamples copyList(ListOfExamples list){
		
		ListOfExamples copy = new ListOfExamples(list);
		Iterator<Example> iterator = list.iterator();
		
		while(iterator.hasNext()){
			copy.add(iterator.next());
		}
		
		return copy;
	}
	
	//RAS - This method takes an example as input and returns a copy of the example.
	public static Example copyExample(Example exampleToCopy){
		
		Example example = new Example(exampleToCopy);
		Iterator<String> iterator = exampleToCopy.iterator();
		
		while(iterator.hasNext()){
			example.add(iterator.next());
		}
		
		return example;
	}
	
	//RAS - This method takes in a ListOfExamples and a feature along with a feature value, and returns the subset of
	//the ListOfExamples with the passed in feature value.
	public static ListOfExamples listWithFeatureValue(ListOfExamples trainExamples, BinaryFeature feature, String value){
		
		ListOfExamples returnList = copyList(trainExamples);
		int count = 0;
		int featIndex;
		Iterator<Example> iterator = trainExamples.iterator();
		Example example;
		String featureValue;
		
		while(iterator.hasNext()){
			
			example = iterator.next();
			featIndex = FeatureIndex(trainExamples, feature.name);
			featureValue = example.get(featIndex);
			
			if ((!(featureValue.equals(value)) && returnList.contains(example))) {
				returnList.remove(example);
				count++;
			}
		}
		
		returnList.numExamples = returnList.numExamples - count;
		return returnList;
	}
	
	//RAS - This feature takes a feature's name as input and returns the features index in the passed in
	//ListOfExamples's ArrayList of features. Using the feature's name takes advantage of our assumed
	//sanitary data set where all features have distinct names. This allows us to not worry about applying
	//the equals operator to BinaryFeatures.
	public static int FeatureIndex(ListOfExamples trainExamples, String feature){
		
		for (int i = 0; i < trainExamples.numFeatures; i++){
			if (feature.equals(trainExamples.features[i].name)){
				return i;
			}
		}
		return 0;
	}
	
	//RAS - This function takes a double as input and returns its log base 2 value.
	public static double logBase2(double number){
		return Math.log(number) / Math.log(2);
	}
	
	//RAS - This is the main recursive tree-building method. It creates a decision tree with a ListOfExamples (which gets smaller on subsequent recursive
	//calls), a tree node (representing the root at the current iteration), as well as a path and the ListOfExamples at the parent node for some auxiliary
	//calculations.
	public static void buildTree(ListOfExamples trainExamples, TreeNode tree, List<BinaryFeature> path, ListOfExamples parentNodeSet){
		
		int numberOfExamples = trainExamples.size();
		
		//one of the termination conditions - there are no examples left
		if ((numberOfExamples == 0) || (path.size() == trainExamples.numFeatures)) {							//If there are no examples left we get the majority label of parent node
			String majorityLabel = MajorityLabel(parentNodeSet);
			tree.addLabel(majorityLabel);
			tree.setNumberOfTrainingExamplesInNode(numberOfExamples);
			return;
		}
		
		//one of the termination conditions - all examples left have the same label
		else if(AllExamplesHaveSameLabel(trainExamples)) {
			Example example = trainExamples.get(0);
			tree.addLabel(example.label);									//doesn't matter which label we 
			tree.setNumberOfTrainingExamplesInNode(numberOfExamples);		//get since they're all the same
			return;
		}
		
		//if we don't hit either termination condition we process the current node
		else {
			
			ArrayList<BinaryFeature> remainingFeatures = RemainingFeatures(trainExamples.features, path);
			int bestFeatureIndex = getBestFeatureIndex(trainExamples, remainingFeatures);
			BinaryFeature bestFeature = trainExamples.features[bestFeatureIndex];
			
			if (!path.contains(bestFeature)){
				path.add(bestFeature);
			}
			
			BinaryFeature copyOfBestFeature = new BinaryFeature(bestFeature.name, bestFeature.firstValue, bestFeature.secondValue);
			tree.addFeature(copyOfBestFeature);		//we use a copy of the feature so it can have its own
													//children and parent
			
			ArrayList<TreeNode> Children = MakeTreeUpdates(tree);
			TreeNode Child;
			
			String[] featureValues = {copyOfBestFeature.firstValue, copyOfBestFeature.secondValue};
			String featureValue = "";
			
			for(int i = 0; i < Children.size(); i++) {
				
				Child = Children.get(i);
				featureValue = featureValues[i];
				
				ListOfExamples listCopy = listWithFeatureValue(trainExamples, copyOfBestFeature, featureValue);
				List<BinaryFeature> pathCopy = copyPath(path);

				buildTree(listCopy, Child, pathCopy, trainExamples);
			}
		}
		return;
	}

	//RAS - This method takes in a tree and returns an ArrayList of size 2 representing the tree node's children. It also updates the tree structure with
	//empty nodes for those children to be populated later
	private static ArrayList<TreeNode> MakeTreeUpdates(TreeNode tree) {
		
		TreeNode leftChild = new TreeNode();
		TreeNode rightChild  = new TreeNode();
		
		tree.setLeftChild(leftChild);
		tree.setRightChild(rightChild);
		
		leftChild.setParent(tree);
		rightChild.setParent(tree);
		
		ArrayList<TreeNode> Children = new ArrayList<TreeNode>();
		
		Children.add(leftChild);
		Children.add(rightChild);
		
		return Children;
	}

	//RAS - This method takes in an array of features and a path (an ArrayList of features), and returns a path. The array of features passed in represents
	//the complete list of features in the feature space, while the path passed in represents the list of features that we want to remove (i.e. the list of
	//features we've already hit somewhere while on our path through the decision tree up to this point.
	private static ArrayList<BinaryFeature> RemainingFeatures(BinaryFeature[] features, List<BinaryFeature> featuresToRemove) {

		ArrayList<BinaryFeature> remainingFeatures = new ArrayList<BinaryFeature>();
		BinaryFeature feature;
		int i;
		int numberOfFeatures = features.length;
		
		for (i = 0; i < numberOfFeatures; i++) {
			
			feature = features[i];
			if (!(featuresToRemove.contains(feature))) {
				
				remainingFeatures.add(feature);
			}
		}

		return remainingFeatures;
	}
	
	//RAS - This method takes a ListOfExamples as input and returns a string corresponding to the output label that the majority of them have.
	//If the examples passed in are evenly split between the first and second possible output value, we return the second value. Just because.
	//This method is used when deciding what label to apply to a leaf node with no children having one of the node's feature values.
	public static String MajorityLabel(ListOfExamples trainExamples){
		
		Example example;
		Iterator<Example> iterator = trainExamples.iterator();
		Hashtable<String, Integer> labelAmounts = new Hashtable<String, Integer>();
		int currentLabelAmount;
		String currentLabel;
		String majorityLabel;
		
		String firstValue = trainExamples.outputLabel.firstValue;
		String secondValue = trainExamples.outputLabel.secondValue;
		labelAmounts.put(firstValue, 0);
		labelAmounts.put(secondValue, 0);
		
		while(iterator.hasNext()){
			example = iterator.next();
			currentLabel = example.label;
			currentLabelAmount = labelAmounts.get(currentLabel) + 1;
			labelAmounts.put(example.label, currentLabelAmount);
			}

		if (labelAmounts.get(firstValue) > labelAmounts.get(secondValue)) {
			majorityLabel = firstValue;
		}
		
		else {
			majorityLabel = secondValue;
			}
		
		return majorityLabel;
	}
	
	//RAS - This method takes in a ListOfExamples and returns whether all of the examples passed in have the same output label,
	//e.g. if all examples passed in have a label of "died", this method returns false; if at least one examples has an output
	//label of "lived" and at least one example has an output label of "died", this method returns true.
	public static boolean AllExamplesHaveSameLabel(ListOfExamples trainExamples){
		
		Example example;
		Iterator<Example> iterator = trainExamples.iterator();
		String labelValue = "";
		boolean allTheSame = true;
		
		if (trainExamples.size() == 1) {
			return allTheSame;
		}
		
		while(iterator.hasNext()){
			example = iterator.next();
			
			if (labelValue == "") {
				labelValue = example.label;
			}
			
			if (!(labelValue.equals(example.label))) {
				allTheSame = false;
				break;
			}
		}

		return allTheSame;
	}
	
	//RAS - This method takes in a path and returns a copy of the path. 
	public static List<BinaryFeature> copyPath(List<BinaryFeature> path){
		
		List<BinaryFeature> copy = new ArrayList<BinaryFeature>();
		
		for (int i = 0; i < path.size(); i++){
			copy.add(i, path.get(i));
		}
		
		return copy;
	}
	
	//RAS - This method takes a decision tree root node and ListOfExamples as input and tests the decision
	//tree on each example in the passed in ListOfExamples. It stores the prediction that the decision tree
	//makes on each example in a Hashtable, which this method returns.
	public static Hashtable<Example, String> TestTree(TreeNode root, ListOfExamples testSet){
		
		Hashtable<Example, String> predictions = new Hashtable<Example, String>();
		Example example;
		TreeNode placeInTree;
		TreeNode leftChild;
		TreeNode rightChild;
		int featIndex = 0;
		int numberOfExamples = testSet.numExamples;
		String featureName = "";
		String firstValue = "";
		String secondValue = "";
		String exampleFeatureValue = "";
		String prediction = "";
		
		for (int i = 0; i < numberOfExamples; i++){
			
			example = testSet.get(i);
			placeInTree = root;
			while (placeInTree.getFeature() != null) {
				
				featureName = placeInTree.getFeature().name;
				leftChild = placeInTree.getLeftChild();
				rightChild = placeInTree.getRightChild();
				
				firstValue = placeInTree.getFeature().firstValue;
				secondValue = placeInTree.getFeature().secondValue;
				featIndex = FeatureIndex(testSet, featureName);
				exampleFeatureValue = example.get(featIndex);
				
				if (exampleFeatureValue.equals(firstValue)){
					placeInTree = leftChild;
				}
				else if (exampleFeatureValue.equals(secondValue)) {
					placeInTree = rightChild;
				}
			}
			
			prediction = placeInTree.getLabel();
			predictions.put(example, prediction);
		}
		
		return predictions;
	}
	
	//RAS - print the decision tree we generated earlier in a readable format.
	public static void PrintTree(int spacesToIndent, TreeNode node){
		
		BinaryFeature feature = node.getFeature();
		TreeNode leftChild = node.getLeftChild();
		TreeNode rightChild = node.getRightChild();
		String printString = "";
		
		if ((leftChild == null) && (rightChild == null)) {
			
			printString = node.getLabel() + " (" + Integer.toString(node.getNumberOfTrainingExamplesInNode()) + ")";
			System.out.print(printString);
			System.out.println();
		}
		
		else {
			
			String featureName = feature.name;
			
			System.out.println();
			for (int i = 0; i < spacesToIndent; i++){
				System.out.print(" ");
			}
			
			System.out.print(featureName + " = " + feature.secondValue + " : ");
			
			PrintTree(spacesToIndent + 3, rightChild);
			
			for (int i = 0; i < spacesToIndent; i++){
				System.out.print(" ");
			}
			
			System.out.print(featureName + " = " + feature.firstValue+ " : ");
			PrintTree(spacesToIndent + 3, leftChild);
		}
	}
	
//////////////////////////////////////////////////////////////////////////////////////////////////
}

// This class, an extension of ArrayList, holds an individual example.
// The new method PrintFeatures() can be used to
// display the contents of the example. 
// The items in the ArrayList are the feature values.
@SuppressWarnings("serial")
class Example extends ArrayList<String>
{
  // The name of this example.
  public String name;  

  // The output label of this example.
  public String label;

  // The data set in which this is one example.
  public ListOfExamples parent;  

  // Constructor which stores the dataset which the example belongs to.
  public Example(ListOfExamples parent) {
    this.parent = parent;
  }
  
  public Example(Example example) {
	  	this.name = example.name;
	  	this.label = example.label;
	    this.parent = example.parent;
	  }

  // Print out this example in human-readable form.
  public void PrintFeatures()
  {
    System.out.print("Example " + name + ",  label = " + label + "\n");
    for (int i = 0; i < parent.getNumberOfFeatures(); i++)
    {
      System.out.print("     " + parent.getFeatureName(i)
                       + " = " +  this.get(i) + "\n");
    }
  }

  // Adds a feature value to the example.
  public void addFeatureValue(String value) {
    this.add(value);
  }

  public String getName() {
    return name;
  }
  
  public void setName(String name) {
	    this.name = name;
	  }

  public String getLabel() {
    return label;
  }

  public void setLabel(String label) {
    this.label = label;
  }
  
  @Override
  public boolean equals(Object obj) {
	    if (obj == null) return false;
	    if (obj == this) return true;
	    if (!(obj instanceof BinaryFeature))return false;
	    BinaryFeature convertedObj = (BinaryFeature) obj;
	    String nameString1 = this.name;
	    String nameString2 = convertedObj.name;
	    boolean result = (nameString1.equals(nameString2));
	    return result;
  }
  

}

/* This class holds all of our examples from one dataset
   (train OR test, not BOTH).  It extends the ArrayList class.
   Be sure you're not confused.  We're using TWO types of ArrayLists.  
   An Example is an ArrayList of feature values, while a ListOfExamples is 
   an ArrayList of examples. Also, there is one ListOfExamples for the 
   TRAINING SET and one for the TESTING SET. 
*/
@SuppressWarnings("serial")
class ListOfExamples extends ArrayList<Example>
{
  // The name of the dataset.
  public String nameOfDataset = "";

  // The number of features per example in the dataset.
  public int numFeatures = -1;

  // An array of the parsed features in the data.
  public BinaryFeature[] features;

  // A binary feature representing the output label of the dataset.
  public BinaryFeature outputLabel;

  // The number of examples in the dataset.
  public int numExamples = -1;

  public ListOfExamples() {} 
  
  //RAS - Constructor that copies a list of examples
  public ListOfExamples(ListOfExamples list){
	  this.nameOfDataset = list.nameOfDataset;
	  this.numFeatures = list.numFeatures;
	  this.features = list.features;
	  this.outputLabel = list.outputLabel;
	  this.numExamples = list.numExamples;
	}
  
  // Print out a high-level description of the dataset including its features.
  public void DescribeDataset()
  {
    System.out.println("Dataset '" + nameOfDataset + "' contains "
                       + numExamples + " examples, each with "
                       + numFeatures + " features.");
    System.out.println("Valid category labels: "
                       + outputLabel.getFirstValue() + ", "
                       + outputLabel.getSecondValue());
    System.out.println("The feature names (with their possible values) are:");
    for (int i = 0; i < numFeatures; i++)
    {
      BinaryFeature f = features[i];
      System.out.println("   " + f.getName() + " (" + f.getFirstValue() +
			 " or " + f.getSecondValue() + ")");
    }
    System.out.println();
  }

  // Print out ALL the examples.
  public void PrintAllExamples()
  {
    System.out.println("List of Examples\n================");
    for (int i = 0; i < size(); i++)
    {
      Example thisExample = this.get(i);  
      thisExample.PrintFeatures();
    }
  }

  // Print out the SPECIFIED example.
  public void PrintThisExample(int i)
  {
    Example thisExample = this.get(i); 
    thisExample.PrintFeatures();
  }

  // Returns the number of features in the data.
  public int getNumberOfFeatures() {
    return numFeatures;
  }

  // Returns the name of the ith feature.
  public String getFeatureName(int i) {
    return features[i].getName();
  }

  // Takes the name of an input file and attempts to open it for parsing.
  // If it is successful, it reads the dataset into its internal structures.
  // Returns true if the read was successful.
  public boolean ReadInExamplesFromFile(String dataFile) {
    nameOfDataset = dataFile;

    // Try creating a scanner to read the input file.
    Scanner fileScanner = null;
    try {
      fileScanner = new Scanner(new File(dataFile));
    } catch(FileNotFoundException e) {
      return false;
    }

    // If the file was successfully opened, read the file
    this.parse(fileScanner);
    return true;
  }

  /**
   * Does the actual parsing work. We assume that the file is in proper format.
   *
   * @param fileScanner a Scanner which has been successfully opened to read
   * the dataset file
   */
  public void parse(Scanner fileScanner) {
    // Read the number of features per example.
    numFeatures = Integer.parseInt(parseSingleToken(fileScanner));

    // Parse the features from the file.
    parseFeatures(fileScanner);

    // Read the two possible output label values.
    String labelName = "output";
    String firstValue = parseSingleToken(fileScanner);
    String secondValue = parseSingleToken(fileScanner);
    outputLabel = new BinaryFeature(labelName, firstValue, secondValue);

    // Read the number of examples from the file.
    numExamples = Integer.parseInt(parseSingleToken(fileScanner));

    parseExamples(fileScanner);
  }

  /**
   * Returns the first token encountered on a significant line in the file.
   *
   * @param fileScanner a Scanner used to read the file.
   */
  private String parseSingleToken(Scanner fileScanner) {
    String line = findSignificantLine(fileScanner);

    // Once we find a significant line, parse the first token on the
    // line and return it.
    @SuppressWarnings("resource")
	Scanner lineScanner = new Scanner(line);
    return lineScanner.next();
  }

  /**
   * Reads in the feature metadata from the file.
   * 
   * @param fileScanner a Scanner used to read the file.
   */
  private void parseFeatures(Scanner fileScanner) {
    // Initialize the array of features to fill.
    features = new BinaryFeature[numFeatures];

    for(int i = 0; i < numFeatures; i++) {
      String line = findSignificantLine(fileScanner);

      // Once we find a significant line, read the feature description
      // from it.
      @SuppressWarnings("resource")
	Scanner lineScanner = new Scanner(line);
      String name = lineScanner.next();
      @SuppressWarnings("unused")
	String dash = lineScanner.next();  // Skip the dash in the file.
      String firstValue = lineScanner.next();
      String secondValue = lineScanner.next();
      features[i] = new BinaryFeature(name, firstValue, secondValue);
    }
  }

  private void parseExamples(Scanner fileScanner) {
    // Parse the expected number of examples.
    for(int i = 0; i < numExamples; i++) {
      String line = findSignificantLine(fileScanner);
      @SuppressWarnings("resource")
	Scanner lineScanner = new Scanner(line);

      // Parse a new example from the file.
      Example ex = new Example(this);

      String name = lineScanner.next();
      ex.setName(name);

      String label = lineScanner.next();
      ex.setLabel(label);
      
      // Iterate through the features and increment the count for any feature
      // that has the first possible value.
      for(int j = 0; j < numFeatures; j++) {
	String feature = lineScanner.next();
	ex.addFeatureValue(feature);
      }

      // Add this example to the list.
      this.add(ex);
    }
  }

  /**
   * Returns the next line in the file which is significant (i.e. is not
   * all whitespace or a comment.
   *
   * @param fileScanner a Scanner used to read the file
   */
  private String findSignificantLine(Scanner fileScanner) {
    // Keep scanning lines until we find a significant one.
    while(fileScanner.hasNextLine()) {
      String line = fileScanner.nextLine().trim();
      if (isLineSignificant(line)) {
	return line;
      }
    }

    // If the file is in proper format, this should never happen.
    System.err.println("Unexpected problem in findSignificantLine.");

    return null;
  }

  /**
   * Returns whether the given line is significant (i.e., not blank or a
   * comment). The line should be trimmed before calling this.
   *
   * @param line the line to check
   */
  private boolean isLineSignificant(String line) {
    // Blank lines are not significant.
    if(line.length() == 0) {
      return false;
    }

    // Lines which have consecutive forward slashes as their first two
    // characters are comments and are not significant.
    if(line.length() > 2 && line.substring(0,2).equals("//")) {
      return false;
    }

    return true;
  }
}

/**
 * Represents a single binary feature with two String values.
 */
class BinaryFeature {
  public String name;
  public String firstValue;
  public String secondValue;

  public BinaryFeature(String name, String first, String second) {
    this.name = name;
    firstValue = first;
    secondValue = second;
  }

  public String getName() {
    return name;
  }

  public String getFirstValue() {
    return firstValue;
  }

  public String getSecondValue() {
    return secondValue;
  }

}

class Utilities
{
  // This method can be used to wait until you're ready to proceed.
  public static void waitHere(String msg)
  {
    System.out.print("\n" + msg);
    try { System.in.read(); }
    catch(Exception e) {} // Ignore any errors while reading.
  }
}

class TreeNode {
	
	private BinaryFeature feature;
	private String label;
	private List<BinaryFeature> path;
	private TreeNode leftChild;
	private TreeNode rightChild;
	private TreeNode parent;
	private int numberOfTrainingExamplesInNode;
	
	public TreeNode(BinaryFeature feature, String option1, String option2) {
		
		this.feature = feature;
		leftChild = null;
		rightChild = null;
		parent = null;
	}
	
	public TreeNode(){
		
		this.feature = null;
		leftChild = null;
		rightChild = null;
		parent = null;
	}
	
	public void setParent(TreeNode parent){
		this.parent = parent;
	}
	
	public TreeNode getParent(){
		return parent;
	}
	
	public void setLeftChild(TreeNode left){
		this.leftChild = left;
	}
	
	public void setRightChild(TreeNode right){
		this.rightChild = right;
	}
	
	public void setLeftChild(BinaryFeature feature, String option1, String option2,
			List<BinaryFeature> path){
		leftChild = new TreeNode(feature, option1, option2);
		path.add(feature);
	}
	
	public void setRightChild(BinaryFeature feature, String option1, String option2,
			List<BinaryFeature> path){
		rightChild = new TreeNode(feature, option1, option2);
		path.add(feature);
	}
	
	public List<BinaryFeature> getPath(){
		return path;
	}
	
	public void addFeature(BinaryFeature feature){
		this.feature = feature;
	}
	
	public void addLabel(String label) {
		this.label = label;
	}
	
	public BinaryFeature getFeature(){
		return feature;
	}
	
	public TreeNode getLeftChild(){
		return this.leftChild;
	}
	
	public TreeNode getRightChild(){
		return this.rightChild;
	}
	
	public String getLabel(){
		return this.label;
	}

	public int getNumberOfTrainingExamplesInNode() {
		return numberOfTrainingExamplesInNode;
	}

	public void setNumberOfTrainingExamplesInNode(int numberOfTrainingExamplesInNode) {
		this.numberOfTrainingExamplesInNode = numberOfTrainingExamplesInNode;
	}
	
}