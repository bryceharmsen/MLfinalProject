package ml;

public class App {
    public static void main(String[] args) throws Exception {
        //run with 'java -Dfile.encoding=UTF-8 @/tmp/cp_vngd4z7nsodb9j8f4wl3xwie.argfile ml.App ./arffs/ ./results/'
        MLP mlp = new MLP(args[0], args[1]);
        mlp.run();
    }
}