import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.io.File;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
// import java.nio.charset.StandardCharsets;
// import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import net.imagej.Dataset;
import net.imagej.DefaultDataset;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imagej.tensorflow.Tensors;
import net.imagej.tensorflow.GraphBuilder;
import net.imagej.ops.OpService;
import net.imagej.ops.convert.clip.ClipRealTypes;
import net.imagej.ops.convert.RealTypeConverter;
import net.imagej.axis.CalibratedAxis;

import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
// import net.imglib2.converter.Converter;
import net.imglib2.converter.RealFloatConverter;
// import net.imglib2.converter.RealUnsignedShortConverter;
// import net.imglib2.converter.RealUnsignedByteConverter;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.real.FloatType;
// import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.util.Intervals;
import net.imglib2.img.Img;
import net.imagej.ImgPlus;
// import net.imglib2.img.ImgView;
// import net.imglib2.view.IntervalView;
// import net.imglib2.view.Views;
import net.imglib2.img.ImgFactory; 
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.display.imagej.ImageJFunctions;
import org.scijava.ItemIO;
import org.scijava.command.Command;							//comment this out when finished
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import ij.IJ;
import ij.ImagePlus;
import ij.process.*;
import ij.ImageStack;
import ij.gui.GenericDialog;
import ij.WindowManager;

import java.text.DecimalFormat;
import ij.plugin.RGBStackMerge;
// import ij.plugin.filter.PlugInFilter;
import ij.plugin.PlugIn;
import ij.io.FileSaver;

import org.tensorflow.*;   //kind of expensive but I'm lazy
import io.scif.services.DatasetIOService;

/**
 * Plugin to run the SISR transform with a pre-built model
 *
 * @author Ronald Gao
 */

// @Plugin(type = Command.class, menuPath = "Analyze>SISR_transform_net")
public class SISR_transform_net<T extends RealType<T> & NativeType<T>> implements PlugIn{

	// -- Possible command options --
	private static final String SINGLE = "Single";
	private static final String BATCH = "Batch";

	// -- Image type constants --
	private static final String PNG = ".png";
	private static final String JPEG = ".jpeg";

	// -- Needed services --

	@PARAMETER doesnt work for plugins!
	@Parameter													// For opening images, datasetService is deprecated
    private  DatasetIOService datasetIOService;				

   	@Parameter													// logging for imagej
	private  LogService log;									

 	@Parameter													// OpService provides tons of operations to perform
    private  OpService ops;				

 	@Parameter
    private DatasetService dataset_Service;

	// @Parameter(type = ItemIO.OUTPUT)					// DEBUGGING: how image is input 
	// 	private Img<T> s_inputImage;

	// @Parameter(type = ItemIO.OUTPUT)					// DEBUGGING: what kind of image is output from model
	// 	private Img<FloatType> outputImage;

	// @Parameter(type = ItemIO.OUTPUT)					// DEBUGGING: image result from clipping
	// 	private Img<T> clippedImage;

	// @Parameter(type = ItemIO.OUTPUT)					// DEBUGGING: what kind of image is loaded
	// 	private Img<FloatType> loadedImage;

//Global Constants

   // private static final String PARAMETERS_FILENAME = "PARAMETERS.txt";

    public String inputDirectory;

    public String outputDirectory;

    public String modelDirectory;

    public String outputType;

    public String outputMethod;
	
//Global Variables
   // private static Map<String, String> ParameterMap;		//don't forget to convert if the value you want is not a String!


//##################################################  Setup  #######################################################

   /**
    *	DESCRIPTION:
	*		Utility function for something like PARAMETERS.TXT file. Every line is a key and value separater by "="		
	*	If this function returns an something that is not supposed to be a String, don't forget to convert
	*   This assumes the format of the parameters file is like
	*   batch_size = 3   <-- for every line 
	*
    *	INPUT: 
    *		String fileName - the file
    *	RETURN VALUE:
    *		ParameterMap - map of keys and values of the file
    *
   **/
   
  //  private static Map<String, String> createParametersMap(String fileName) throws IOException, FileNotFoundException {
  //  		Map<String,String> ParameterMap = new HashMap<String,String>();
  //  		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));
		// try {

		//     String line;
		//     System.out.println("Here are the parameters: ");
		//     System.out.println("");
		//     while ((line = br.readLine()) != null) {
		// 	    String[] t = line.split("=");
		// 	    ParameterMap.put(t[0].replaceAll("\\s",""), t[1].replaceAll("\\s",""));
		// 	    System.out.println(t[0] + "=" + t[1]);
		// 	}
	 //    // } catch( FileNotFoundException e){
	 //    // 	System.out.println("FileNotFoundException: Did not find a PARAMETERS.txt file or something similar");
	 //    // } catch ( IOException e ){
	 //    // 	System.out.println("IOException caught inside createParametersMap function!");
		// } finally {
		//     br.close();
		// }
		// return ParameterMap;
  //  }

    /**
	*	DESCRIPTION: 
	*		Loads the image from the image directory. Beware that Tensorflow and ImageJ use different image dimensions
	*	This function adjusts the dimensions for you. Assumes the output tensor needed is tf.float32
	*	
	*	INPUT:
	*		Dataset d - the image to be loaded. Dataset is a structure for Imagej. It helps with image processing
	*	RETURN VALUE:	
	*		Tensor - This tensor is already reshaped so it is ready to use for Tensorflow models
   **/
	@SuppressWarnings({ "rawtypes", "unchecked" })
	private static Tensor loadFromImgLib(final Dataset d, Graph g, GraphBuilder b, Integer count) {
		return loadFromImgLib((RandomAccessibleInterval) d.getImgPlus(), g, b, count);
	}
	private static <T extends RealType<T>> Tensor loadFromImgLib(
		final RandomAccessibleInterval<T> image, Graph g, GraphBuilder b, Integer count)
	{
		//Maybe more efficient here
		final RandomAccess<T> source = image.randomAccess();
		final long[] dims = Intervals.dimensionsAsLongArray(image);
		final long[] reshapedDims = new long[] { dims[1], dims[0], dims[2] };

		final ArrayImg<FloatType, FloatArray> dest = ArrayImgs.floats(
			reshapedDims);
		final Cursor<FloatType> destCursor = dest.cursor();
		for (int y = 0; y < dims[1]; y++) {
			source.setPosition(y, 1);
			for (int x = 0; x < dims[0]; x++) {
				source.setPosition(x, 0);
				for (int c = 0; c < dims[2]; c++) {
					destCursor.fwd();
					source.setPosition(c, 2);
					destCursor.get().setReal(source.get().getRealDouble());
				}
			}
		}
		
		//expand dimensions here to match. result should be 1,400,400,3
		// RealFloatConverter<T> converter = new RealFloatConverter<>();
		// Tensor t = Tensors.tensor(Converters.convert(image, converter, new FloatType()), new int[]{ 1, 0, 2 });

		//Unable to find a more efficient way to do this

		// Output placeholder = g.opBuilder("Placeholder", "input").setAttr("dtype", t.dataType()).setAttr("value", t).build().output(0);
		Graph g2 = new Graph();
		final GraphBuilder b2 = new GraphBuilder(g2);
		// final Output input = g2.opBuilder("Const", "input")
		// 	.setAttr("dtype", t.dataType())
		// 	.setAttr("value", t)
		// 	.build().output(0);
		final Output input = b2.constant("input", dest.update(null)
				.getCurrentStorageArray(), reshapedDims);

		final Output output = b2.expandDims(
			input, b2.constant("make_batch" + count,0 ));
		try (Session s = new Session(g2)) {
			return s.runner().fetch(output.op().name()).run().get(0);
		}
		// final Output input = b.expandDims( 
		// 	b.constant("input_" + count.toString(), rai, ),
		// 	b.constant("make_batch_"+ count.toString(), 0));
		// final Output input = b.expandDims(
		// 	b.constant("input_" + count.toString(), dest.update(null).getCurrentStorageArray(), reshapedDims),
		// 	b.constant("make_batch_"+ count.toString(), 0));
		// try (Session s = new Session(g)) {
		// 	return s.runner().fetch(input.op().name()).run().get(0);
		// }
	}

	private boolean showDialogSB() {
		GenericDialog gd = new GenericDialog("Single or Batch Method");

		gd.addChoice("Save as:", new String[]{"Single", "Batch"}, "Single");      

		gd.showDialog();
		if (gd.wasCanceled())
			return false;

		//get entered values	
		outputMethod = gd.getNextChoice();

		//print out the values since the dialog box is so small
		System.out.println("Output Method: " + outputMethod);

		return true;
	}

	private void showDialog() {
		GenericDialog gd = new GenericDialog("SISR_transform_net");

		//get current directory
		final String curr_dir = System.getProperty("user.dir") + "\\";

		// default directory is the current directory
		gd.addStringField("Input Image Directory", curr_dir);
		gd.addStringField("Model Directory", curr_dir + "Models");
		gd.addStringField("Output Directory", curr_dir + "output_imgs");
		//choices of JPEG and PNG, default is PNG
		gd.addChoice("Save as:", new String[]{"JPEG", "PNG"}, "PNG");      

		gd.showDialog();
		// if (gd.wasCanceled())
		// 	return false;

		//get entered values	
		inputDirectory = gd.getNextString() + "\\";
		modelDirectory = gd.getNextString() + "\\";
		outputDirectory = gd.getNextString() + "\\";
		outputType = gd.getNextChoice();

		//print out the values since the dialog box is so small
		System.out.println("Input Directory " + inputDirectory);
		System.out.println("Model Directory: " + modelDirectory);
		System.out.println("Output Directory: " + outputDirectory);
		System.out.println("Output image type: " + outputType);

		//return true;
	}

//##################################################  Main  #######################################################

	public void setup(){
		System.out.println( "Hello World! I'm using tensorflow version " + TensorFlow.version() );
		System.out.println( "Make sure maven is running same version of tensorflow as loaded model!\n ");
	}

    @Override
    public void run(String arg){


    	//Get model
    	//get current directory
		final String curr_dir = System.getProperty("user.dir") + "\\";
		modelDirectory = curr_dir + "Models" + "\\";

		try{
			//Pick Single vs Batch
			showDialogSB();
			
			setup();
			if(outputMethod == SINGLE){
				//##################################### Logisitcs #####################################################

				//Load current open image in imagej
				// ImagePlus image = WindowManager.getCurrentImage();
				// if(image != null)
				// 	image.show();
				// else{
				// 	System.out.println("No Image placed in ImageJ to run model with!");
				// 	// System.exit(1);
				// }
				ImagePlus image = IJ.openImage("http://imagej.net/images/clown.jpg");
				// image.show();

				//Load pre-trained model
				final long loadModelStart = System.nanoTime();  										//start time to load
				SavedModelBundle smb = SavedModelBundle.load( modelDirectory + "2/", "serve");     
				final long loadModelEnd = System.nanoTime();											//end time to load
				System.out.println(String.format(
					"Successfully loaded model in %dms", (loadModelEnd -
						loadModelStart) / 1000000));


				//Create ParameterMap since java doesn't have import txt files like python does
				// ParameterMap = createParametersMap(curr_dir + PARAMETERS_FILENAME);


				//###################################### Done with Logisitcs. Load and test model! ####################
				final long runModelStart = System.nanoTime();	
				
				//start session using the model we just loaded
				Session sess = smb.session();

				//Use the graph we just loaded. don't use new graph since it won't have operations defined
				Graph g = smb.graph();						

				// Java specific util class that I included to build Graph operations
				GraphBuilder b = new GraphBuilder(g);		

				//convert from ImagePlus to Img<T>
				Img<T> inputImage = ImageJFunctions.wrap(image);


				Dataset d_inputImage = dataset_Service.create(inputImage);

				// loads it into 32 bit signed float or FloatType. Shape for test was 400,400,3
				final Tensor inputTensor = loadFromImgLib(d_inputImage, g, b, 0); 
				// Img <T> loadedImage = Tensors.imgFloat(inputTensor, new int[]{ 2, 1, 3, 0});		// to see image from loadFromImgLib	

				//Network at index 0, bilinear at index 1
				Tensor[] results = runNetworkAndBilinear(sess, inputTensor);

				//run twice for the two network results
				
				convertTensorToRGB(results[0], inputImage.firstElement()).show("Network");
				convertTensorToRGB(results[0], inputImage.firstElement()).show("Bilinear");

				//End of running model. Output how long it took
				final long runModelEnd = System.nanoTime();
				System.out.println(String.format("Ran image through model in %dms", //
						(runModelEnd - runModelStart) / 1000000));
				smb.close();
			}
			else //Batch
			{
				System.exit(0);
				// if (!showDialog())
				// 	System.exit(0);
			}
			// //##################################### Logisitcs #####################################################

			// //Load pre-trained model
			// final long loadModelStart = System.nanoTime();  										//start time to load
			// SavedModelBundle smb = SavedModelBundle.load( modelDirectory + "2/", "serve");     
			// final long loadModelEnd = System.nanoTime();											//end time to load
			// log.info(String.format(
			// 	"Successfully loaded model in %dms", (loadModelEnd -
			// 		loadModelStart) / 1000000));


			// //Create ParameterMap since java doesn't have import txt files like python does
			// // ParameterMap = createParametersMap(curr_dir + PARAMETERS_FILENAME);


			// //###################################### Done with Logisitcs. Load and test model! ####################
			// final long runModelStart = System.nanoTime();	
			
			// //start session using the model we just loaded
			// Session sess = smb.session();

			// //Use the graph we just loaded. don't use new graph since it won't have operations defined
			// Graph g = smb.graph();						

			// // Java specific util class that I included to build Graph operations
			// GraphBuilder b = new GraphBuilder(g);		

			// makeOutputDirectory();

			// // get input Images from directory
			// File folder = new File(inputDirectory);
			// File[] listOfFiles = folder.listFiles();
			// Arrays.sort(listOfFiles, new Comparator<File>(){
			// 	@Override
			// 	public int compare(File a, File b){
			// 		int n1 = extractNumber(a.getName());
	  //               int n2 = extractNumber(b.getName());
	  //               return n1 - n2;
			// 	}
			// 	//In case some file inside the given directory does not follow the format "ImageName.ImageType"
			// 	private int extractNumber(String name) {
	  //               int i = 0;
	  //               try {
	  //                   int s = name.indexOf('_')+1;
	  //                   int e = name.lastIndexOf('.');
	  //                   String number = name.substring(s, e);
	  //                   i = Integer.parseInt(number);
	  //               } catch(Exception e) {
	  //                   i = 0; // if filename does not match the format
	  //                          // then default to 0
	  //               }
	  //              	return i;
   //          	}
			// });
			// for(int i = 0; i < listOfFiles.length; i++){
			// 	if(listOfFiles[i].isFile()){
			// 		String fileName = listOfFiles[i].getName();

			// 		Dataset inputDataset = datasetIOService.open(inputDirectory + fileName);
			// 		Img <T> inputImage = (Img<T>) inputDataset.getImgPlus();   						

					// loadFromImgLib();
			// 		Tensor[] results = runNetworkAndBilinear(sess, inputImage);

			// 		//second argument is for inputImage's type. usually uint8
			// 		String fileNameWithoutExtension = fileName.split("\\.(?=[^\\.]+$)")[0];
			// 		File theDir = new File(outputDirectory + fileNameWithoutExtension);
			// 		if(!theDir.exists()){
			// 			try{
			// 		        theDir.mkdir();
			// 		    } 
			// 		    catch(SecurityException se){
			// 		        System.out.println("Unable to create directory: " + outputDirectory + fileName);
			// 		    }        
			// 			saveTensorflowTensorAsImage(network_result, inputImage.firstElement(), theDir.getName() + "\\" + "network");
			// 			saveTensorflowTensorAsImage(bicubic_result, inputImage.firstElement(), theDir.getName() + "\\" + "bicubic");
						// //save the image
						// saveImage(outputRGBImg, fileName);
						// //save the image
						// saveImage(outputRGBImg, fileName);
						// float[][][] contents = result.copyTo(new float[400][400][3]);
						// System.out.println(Arrays.toString(contents));						//to print out the parameters inside
			// 		}else{
			// 			System.out.println(outputDirectory + fileName + " already exists!");
			// 		}

			// 		//finished with this file so log it out
			// 		log.info(fileName);
			// 	}
			// }
			// final long runModelEnd = System.nanoTime();
			// log.info(String.format("Ran image through model in %dms", //
			// 		(runModelEnd - runModelStart) / 1000000));
			// smb.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
   }

   //perform network and bilinear operation on image
   //Returns network result at index 0 and bilinear result at 1
   private Tensor[] runNetworkAndBilinear(Session sess, Tensor inputTensor){
   	    //Run network model
   		Tensor network_result = sess.runner()
			.feed("Placeholder_3", inputTensor)   //(1 400 400 3)
			.fetch("add_35")
			.run().get(0);

		//bicubic test we did in python. Call here
		int[] bilinear_shape = new int[2];
		bilinear_shape[0] = 1000;
		bilinear_shape[1] = 1000;
		Tensor t_bilinear_shape = Tensor.create(new long[] {2}, IntBuffer.wrap(bilinear_shape));
		Tensor bicubic_result = sess.runner()
			.feed("Placeholder_3", inputTensor)   //(1 400 400 3)
			.feed("Placeholder_5", t_bilinear_shape)
			.fetch("ResizeBicubic_1")
			.run().get(0);

		// //to look at the output
		// System.out.println(result.dataType());
		// System.out.println("result shape: " + result.toString());

		return new Tensor[]{network_result, bicubic_result};
   }

	private void makeOutputDirectory(){
		File dir = new File(outputDirectory);
		if(!dir.exists()){
			try{
				dir.mkdir();
			}catch(SecurityException se){
				System.out.println("Unable to create directory: " + outputDirectory);
			}
		}else{
			System.out.println(outputDirectory + " already exists!");
		}
	}

    private void saveImage(ImagePlus imp, String name)
	{
		FileSaver saver = new FileSaver(imp);
		boolean saved_successful = false;
		if(outputType == "PNG")
			saved_successful = saver.saveAsPng(outputDirectory + name + PNG);
		else
			saved_successful = saver.saveAsJpeg(outputDirectory + name + JPEG);
		if (!saved_successful)
			System.out.println("Failed to save image to file: " + outputDirectory);
	}

    public ImagePlus[] splitStackIntoRGBChannels(ImagePlus imp) {

        String sLabel = imp.getTitle();
        String sImLabel = "";
        ImageStack stack = imp.getStack();
        ImagePlus[] channels = new ImagePlus[3];

        int sz = stack.getSize();
        int currentSlice = imp.getCurrentSlice();  // to reset ***

        DecimalFormat df = new DecimalFormat("0000");         // for title

        for(int n=1;n<=sz;++n) {
            imp.setSlice(n);   // activate next slice ***

            // Get current image processor from stack.  What ever is
            // used here should do a COPY pixels from old processor to
            // new. For instance, ImageProcessor.crop() returns copy.
            ImageProcessor ip = imp.getProcessor(); // ***
            ImageProcessor newip = ip.createProcessor(ip.getWidth(),
                                                      ip.getHeight());
            newip.setPixels(ip.getPixelsCopy());

            // Create a suitable label, using the slice label if possible
            sImLabel = imp.getStack().getSliceLabel(n);
            if (sImLabel == null || sImLabel.length() < 1) {
                sImLabel = "slice"+df.format(n)+"_"+sLabel;
            }
            // Create new image corresponding to this slice.
            ImagePlus im = new ImagePlus(sImLabel, newip);
            im.setCalibration(imp.getCalibration());
            channels[n-1] = im;
            
            // Show this image.
            // im.show();
       }

        imp.hide();
        return channels;
    }

    //saves a Tensorflow Tensor back as an Image
    private ImagePlus convertTensorToRGB(Tensor result, T type){
    	//convert back to imagej dimensions from tensorflow dimensions
    	Img<FloatType> outputImage = Tensors.imgFloat(result, new int[] {2,1,3,0});   
		// outputImage = Tensors.imgFloat(result, new int[]{ 1, 0, 2});   //Img<FloatType> 

		//Clip the image from FloatType to any generic type, which usually is uint8: 0-255 
		RealTypeConverter<FloatType, T> clip_op;
		clip_op = (RealTypeConverter<FloatType, T>) ops.op("convert.clip", outputImage.firstElement(), type);
		
		//need to create image with outputImage size which is 1000x1000x3 in this case
		//input image size is 400x400x3
		final ImgFactory<T> img_factory = new ArrayImgFactory<T>();
		//T type of size outputImage. usually uint8
		Img<T> clippedImage = img_factory.create(outputImage, type);	
		
		// perform clip operation on 2nd arg, comes out in 1st arg
		ops.convert().imageType(clippedImage, outputImage, clip_op);   				

		//Clipping not the same as converting to uint8 so do it here
		Img<UnsignedByteType> img = ops.convert().uint8(clippedImage);  			

		//Convert Img<T> to ImagePlus for us to perform stack splitting operations
		ImagePlus final_img = ImageJFunctions.wrap(img, "final_img");				

		//dataset stores the result into array of 3 images so here we separate it. 
		ImagePlus[] RGB_channels = splitStackIntoRGBChannels(final_img);		

		//Merge the separated RGB images so we have one composite image. False means don't keep source images	
		ImagePlus outputRGBImg = RGBStackMerge.mergeChannels(RGB_channels, false);  
		
		return outputRGBImg;
		// Img save_img = ImagePlusAdapter.wrap(outputRGBImg);							//ImagePlus->Img back again! 
		// ImageJFunctions.show(save_img);
			// final ImgPlus s_img = ImagePlusAdapter.wrapImgPlus(final_img);		//<--works for sure but above is just fine
		// Dataset d222 = dataset_Service.create(save_img);
		
    }


    /**
	 * Main method for debugging.
	 *
	 * For debugging, it is convenient to have a method that starts ImageJ, loads
	 * an image and calls the plugin, e.g. after setting breakpoints.
	 *
	 * @param args unused
	 */
    public static void main(String... args) throws IOException{
		// Launch an ImageJ instance
		final ImageJ ij = net.imagej.Main.launch(args);

		// ImagePlus image = WindowManager.getCurrentImage();
		// if(image != null)
		// 	image.show();
		//Launch the SISR transform command
		// ij.command().run(SISR_transform_net.class, true);
		
		//set the plugins.dir property to make the plugin appear in the Plugins menu
		Class<?> clazz = SISR_transform_net.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);

		System.out.println("pluginsDir: " + pluginsDir);

		//run the plugin
		IJ.runPlugIn(clazz.getName(), "");
    }
  
	
}
