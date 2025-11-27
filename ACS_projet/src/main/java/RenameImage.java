import java.io.File;

public class RenameImage {

    public static void main(String[] args) {

        // Path to your dataset (parent folder)
        String datasetPath = "C:/Users/ADMIN/Desktop/Subjects/Courses/IN450_Base de IA/Dataset/";

        File datasetFolder = new File(datasetPath);

        if (!datasetFolder.isDirectory()) {
            System.out.println("Invalid dataset path");
            return;
        }

        // Loop through each label folder
        for (File labelFolder : datasetFolder.listFiles()) {
            if (labelFolder.isDirectory()) {
                renameImagesInFolder(labelFolder);
            }
        }

        System.out.println("Renaming completed.");
    }

    private static void renameImagesInFolder(File folder) {
        File[] files = folder.listFiles((dir, name) ->
                name.toLowerCase().endsWith(".jpg") ||
                name.toLowerCase().endsWith(".jpeg") ||
                name.toLowerCase().endsWith(".png"));

        if (files == null) return;

        System.out.println("Renaming in: " + folder.getName());

        int counter = 1;

        for (File file : files) {

            String extension = "";

            int dotIndex = file.getName().lastIndexOf(".");
            if (dotIndex >= 0) {
                extension = file.getName().substring(dotIndex);
            }

            File newFile = new File(folder, counter + extension);

            // Avoid overwriting if file already exists
            while (newFile.exists()) {
                counter++;
                newFile = new File(folder, counter + extension);
            }

            boolean success = file.renameTo(newFile);

            if (success) {
                System.out.println("Renamed: " + file.getName() + " -> " + newFile.getName());
            } else {
                System.out.println("FAILED: " + file.getName());
            }

            counter++;
        }
    }
}
