<template>
  <div class="Folders">
    <div class="operationBoX">
      <el-button class="operationBtn uploadBtn"type="primary" icon="el-icon-upload2" @click="uploadDialogVisible = true">Upload</el-button>
      <el-button-group class="operationBtn operationBtnGrp">
        <el-button  type="primary" icon="el-icon-video-play" @click="handleRecognizeClick">Recognize</el-button>
<!--        <el-button  type="primary" icon="el-icon-circle-plus-outline">新建记忆集</el-button>-->
      </el-button-group>
      <el-input class="searchInput"
                placeholder="search"
                suffix-icon="el-icon-search"
                v-model="searchContent">
      </el-input>

      <!-- Upload Image Dialog -->
      <el-dialog :visible.sync="uploadDialogVisible" title="Upload Image" width="30%">
        <el-form :model="form">
          <el-form-item label="Picture URL">
            <el-input v-model="form.pictureFilePath" placeholder="Enter picture URL"></el-input>
          </el-form-item>
          <el-form-item label="Picture Name">
            <el-input v-model="form.pictureName" placeholder="Enter picture name"></el-input>
          </el-form-item>
        </el-form>
        <span slot="footer" class="dialog-footer">
        <el-button @click="uploadDialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="handleUpload">Confirm</el-button>
      </span>
      </el-dialog>

      <!-- Recognize Image Dialog -->
      <el-dialog :visible.sync="recognizeDialogVisible"
                 :before-close="handleClose"
                 v-loading="isRecognizeLoading"
                 element-loading-text="Recognizing...Please wait">
        <span slot="title">Recognize Pictures</span>

        <el-form ref="form">
          <el-form-item v-for="row in selectedRows" :key="row.pictureName">
            <span>{{ row.pictureName }}</span>
          </el-form-item>
        </el-form>

        <span slot="footer" class="dialog-footer">
        <el-button @click="recognizeDialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="handleRecognizeConfirm">Confirm</el-button>
      </span>
      </el-dialog>

    </div>
    <div class="tableBox">
      <router-view></router-view>
    </div>
  </div>

</template>

<script>

import {updatePictures, uploadPictures} from "@/api/pictures";
import {postToClassifier} from "@/api/classify";

export default {
  name:'Pictures',
  data(){
    return{
      searchContent:'',
      selectedRows: [],
      uploadDialogVisible:false,
      recognizeDialogVisible:false,
      isRecognizeLoading: false,
      form: {
        pictureName: '',
        pictureFilePath: '',
        pictureOwnerId: JSON.parse(sessionStorage.getItem('user')).userId
      }
    }
  },
  methods: {
    handleClose(done) {
      this.recognizeDialogVisible = false;
      done();
    },
    updateTable(){
      this.$bus.$emit('update table', 0);
    },
    handleRecognizeClick(){
      if (this.selectedRows.length == 0) {
        alert("Please select pictures you want to recognize first!")
      }
      else {
        this.recognizeDialogVisible = true
      }
    },
    async handleRecognizeConfirm() {
      this.isRecognizeLoading = true;
      // Assuming you want to send only the pictureNames to the classifier
      const pictures = this.selectedRows;

      try {
        const labeledPictures = await postToClassifier(pictures);
        // Handle the returned labels as needed
        await updatePictures(labeledPictures)
        await this.updateTable()
      } catch (error) {
        console.error("Error sending data to classifier:", error);
      } finally {
        this.isRecognizeLoading = false;
        this.recognizeDialogVisible = false;
      }
    },
    async handleUpload() {
      try {
        // Show $confirm message box
        await this.$confirm('Are you sure you want to upload this image?', 'Confirm Upload', {
          confirmButtonText: 'Yes',
          cancelButtonText: 'No',
          type: 'warning',
        });

        await uploadPictures(this.form)

        this.$message({
          type: 'success',
          message: 'Image uploaded successfully!',
        });

      } catch (err) {
        // User cancelled the $confirm message box
        this.$message({
          type: 'info',
          message: 'Image upload cancelled',
        });
      } finally {
        // Reset the form and close the dialog
        this.form.pictureFilePath = '';
        this.form.pictureName = '';
        this.uploadDialogVisible = false;
      }
      this.updateTable()
    },
  },
  mounted() {
    this.$router.replace({name:'PicturesTable'})
    this.$bus.$on('selected pics', (data)=>{
      this.selectedRows=data
    })
  }
}
</script>

<style scoped>

.Folders {
  height: 100%;
}
.operationBoX {
  height: 15%;
  display: flex;
}

.tableBox {
  height: 85%;
}


.operationBtn{
  position: relative;
  margin-top: auto;
  margin-bottom: auto;
}

.uploadBtn {
  left: 50px;
}

.operationBtnGrp {
  left: 100px;
}

.searchInput {
  position: relative;
  margin-top: auto;
  margin-bottom: auto;
  width: 200px;
  left: 850px;
}


</style>