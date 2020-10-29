

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import javax.imageio.ImageIO;

public class KMeans {

    public static void main(String [] args){
        if (args.length < 3){
            System.out.println("Usage: Kmeans <input-image> <k> <output-image>");
            return;
        }
        try{
            BufferedImage originalImage = ImageIO.read(new File(args[0]));
            int k=Integer.parseInt(args[1]);
            BufferedImage kmeansJpg = kmeans_helper(originalImage,k);
//            System.out.println(args[0]);
//            System.out.println(args[1]);
//            System.out.println(args[2]);
            ImageIO.write(kmeansJpg, "jpg", new File(args[2]));

        }catch(IOException e){
            System.out.println(e.getMessage());
        }
    }

    private static BufferedImage kmeans_helper(BufferedImage originalImage, int k){
        int w=originalImage.getWidth();
        int h=originalImage.getHeight();
        BufferedImage kmeansImage = new BufferedImage(w,h,originalImage.getType());
        Graphics2D g = kmeansImage.createGraphics();
        g.drawImage(originalImage, 0, 0, w,h , null);
        // Read rgb values from the image
        int[] rgb=new int[w*h];
        int count=0;
        for(int i=0;i<w;i++){
            for(int j=0;j<h;j++){
                rgb[count++]=kmeansImage.getRGB(i,j);
            }
        }
        // Call kmeans algorithm: update the rgb values
        kmeans(rgb,k);

        // Write the new rgb values to the image
        count=0;
        for(int i=0;i<w;i++){
            for(int j=0;j<h;j++){
                kmeansImage.setRGB(i,j,rgb[count++]);
            }
        }
        return kmeansImage;
    }

    // Your k-means code goes here
    // Update the array rgb by assigning each entry in the rgb array to its cluster center

    private static void kmeans(int[] rgb, int k){

        int size = rgb.length;
        int[] clusterCenters = new int[k];

        Random rd = new Random();
        for (int i=0; i<clusterCenters.length;i++)
        {
            clusterCenters[i]= rgb[rd.nextInt(size)];
        }
//        int[] clusterCe

        HashMap<Integer,Integer> clusterMap = new HashMap<Integer,Integer>();
        for(int iteration=0;iteration<100;iteration++) {

            int[] auxiliaryCcBlue= new int[k];
            int[] auxiliaryCcGreen= new int[k];
            int[] auxiliaryCcRed= new int[k];
            int[] auxiliaryCcCount= new int[k];

            for (int point = 0; point < rgb.length; point++) {
                int[] cluster_distance = new int[k];
                int pBlue = rgb[point] & 0xff;
                int pGreen = (rgb[point] & 0xff00) >> 8;
                int pRed = (rgb[point] & 0xff0000) >> 16;

                for (int cluster = 0; cluster < k; cluster++) {
                    int cBlue = clusterCenters[cluster] & 0xff;
                    int cGreen = (clusterCenters[cluster] & 0xff00) >> 8;
                    int cRed = (clusterCenters[cluster] & 0xff0000) >> 16;

                    cluster_distance[cluster] = Math.abs(cBlue - pBlue) + Math.abs(cGreen - pGreen) + Math.abs(cRed - pRed);
                }

                int max = Integer.MAX_VALUE;
                int index = -1;
                for (int i = 0; i < cluster_distance.length; i++) {
                    if (cluster_distance[i] < max) {
                        max = cluster_distance[i];
                        index = i;
                    }
                }
                clusterMap.put(point, index);
                auxiliaryCcBlue[index]+=pBlue;
                auxiliaryCcGreen[index]+=pGreen;
                auxiliaryCcRed[index]+=pRed;
                auxiliaryCcCount[index] +=1;
            }

            for(int i=0;i<clusterCenters.length;i++)
            {
                int avgBlue = auxiliaryCcBlue[i]/auxiliaryCcCount[i];
                int avgGreen = auxiliaryCcGreen[i]/auxiliaryCcCount[i];
                int avgRed = auxiliaryCcRed[i]/auxiliaryCcCount[i];

//                clusterCenters[i]=
            }
        }
    }



}
